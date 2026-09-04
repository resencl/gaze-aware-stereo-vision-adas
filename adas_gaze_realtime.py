# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Claudia Resendiz-Jurado and contributors

"""
Real-time ADAS gaze and distance estimation

Includes:
- Simultaneous capture from two OAK-D cameras.
- Gaze estimation and comparison with manually clicked ground truth.
- Pixel error, NSS, and angular gaze error in degrees.
- End-to-end application FPS and effective FPS for each camera.
- ADAS counts, TTC, warnings, and latency measurements.
- Optional storage of unlabeled LR frames for preparing a YOLO dataset; this
  executable does not calculate precision, recall, or mAP.
"""
import cv2
import depthai as dai
import numpy as np
import time
import torch
import torchvision.transforms as transforms
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
import sys
try:
    import winsound
except ImportError:
    winsound = None

try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ Ultralytics no instalado. Ejecuta: pip install ultralytics")

# Import shared functions from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from gaze_models import (
    _extract_sd, get_gaze_model
)
from gaze_geometry import (
    build_face_alignment_transform,
    model_angles_to_direction,
    validate_proper_rotation,
)

class GazeAwareADAS:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.frame_width = 640
        self.frame_height = 360
        self.K_lr = np.array([[465.0, 0.0, 320.0], [0.0, 465.0, 180.0], [0.0, 0.0, 1.0]])
        self.K_pro = np.array([[465.0, 0.0, 320.0], [0.0, 465.0, 180.0], [0.0, 0.0, 1.0]])
        self.R_lr_from_pro = np.eye(3, dtype=np.float64)
        self.T_lr_from_pro_m = np.zeros(3, dtype=np.float64)
        self.extrinsic_rmse_mm = None
        self.load_extrinsic_calibration(args.extrinsics)
        # Existing end-to-end convention validated with the published results:
        # model directions are interpreted directly in the PRO camera frame.
        self.R_pro_from_model = np.eye(3, dtype=np.float64)
        self.model_frame_status = "existing-validated-convention"
        self.asset_paths = self.validate_required_models()
        self.device_lr = None
        self.device_pro = None
        self.current_model = None
        self.current_model_name = "openvino_fallback"
        self.current_model_config = {}
        self.model_names = ["resnet101_v2"]
        self.model_index = 0

        # Optional manual-validation data.
        self.test_samples = []
        self.current_session = []
        self.gaze_history = deque(maxlen=2)  # Light smoothing preserves responsiveness.
        
        self.waiting_for_click = False
        self.is_paused = False
        self.last_gaze_point = (320, 180)
        self.gaze_valid = False
        self.last_projection_status = "not-evaluated"
        self.click_point = None
        self.error_distance = 0
        self.last_frame = None

        # Live FPS for the full processing loop and each camera stream.
        self.current_fps = 0.0
        self.avg_fps = 0.0
        self.fps_history = deque(maxlen=30)
        self._fps_last_time = time.perf_counter()
        self._fps_last_print = time.perf_counter()
        self.fps_frame_count = 0

        self.camera_fps = {'LR': 0.0, 'Pro': 0.0}
        self.camera_avg_fps = {'LR': 0.0, 'Pro': 0.0}
        self.camera_fps_history = {'LR': deque(maxlen=30), 'Pro': deque(maxlen=30)}
        self._camera_last_ts = {'LR': None, 'Pro': None}
        self.camera_frame_count = {'LR': 0, 'Pro': 0}
        self._camera_first_wall_time = {'LR': None, 'Pro': None}
        self._camera_last_wall_time = {'LR': None, 'Pro': None}
        
        # Depth and parallax state.
        self.last_depth_frame = None
        self.last_pro_depth_frame = None
        self.face_origin_pro = None
        self.current_depth_m = 0.0
        self.model_stats = defaultdict(lambda: {
            'samples': 0,
            'total_error': 0,
            'errors': [],
            'angular_errors': [],
            'click_times': [],
            'nss_scores': []
        })
        
        # Latency and warning metrics.
        self.pipeline_latencies = []       # Capture-to-warning latency per frame in ms.
        self.alarm_events = []
        self.alarm_trigger_count = 0
        self.total_frames_processed = 0
        self.frames_with_danger = 0
        self.risk_condition_frames = 0
        self.alarm_positive_frames = 0
        self.ttc_positive_frames = 0
        self.face_candidate_frames = 0
        self.landmark_valid_frames = 0
        self.rgbd_face_depth_valid_frames = 0
        
        # Precompute the DR(eye)VE-style heatmap.
        self.hm_size = 140
        self.hm_radius = self.hm_size // 2
        x = np.arange(0, self.hm_size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = self.hm_radius
        sigma = 22.0
        gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        
        # Cut off the gaussian at 3 sigma to avoid a bounding box artifact
        gaussian[gaussian < 0.05] = 0.0
        
        heatmap_gray = np.uint8(255 * gaussian)
        self.heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        self.heatmap_alpha = gaussian[:, :, np.newaxis] * 0.6  # 60% max opacity
        
        # YOLO segmentation and warning state.
        self.yolo_model = None
        self.current_looking_at = "Fondo"
        self.alarm_active = False
        self.last_alarm_time = 0
        self.alarm_cooldown = 0.5  # Seconds between audible warnings.
        
        # Optical flow and RANSAC vehicle-motion estimation.
        self.prev_gray = None
        self.prev_pts = None
        self.feature_params = dict(maxCorners=400, qualityLevel=0.01, minDistance=20, blockSize=3)
        self.lk_params = dict(winSize=(11, 11), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.vehicle_moving = False
        self.motion_magnitude = 0.0
        
        # Object tracking and Time-to-Collision (TTC).
        self.tracked_objects = {}
        self.next_track_id = 0
        self.ttc_threshold = 2.0
        self.depth_history_length = 5
        self.previous_frame_input_time = None

        # Accumulated session counters for reports
        self.moving_frames_count = 0
        self.total_detections_while_moving = 0
        self.ttc_positive_records = 0
        self.unattended_ttc_positive_records = 0
        
        self.current_counts = {
            'reliable_stereo': 0,
            'far_stereo': 0,
            'persona': 0,
            'vehiculo': 0,
            'camion_bus': 0,
            'bicicleta': 0,
            'motocicleta': 0,
            'lane_total': 0,
            'lane_unseen': 0,
            'lane_critical_ttc': 0
        }
        if 'YOLO' in globals():
            print("🔍 Cargando modelo YOLOv8-seg...")
            try:
                self.yolo_model = YOLO(str(self.asset_paths["yolo_model"]))
                print("✅ YOLOv8-seg cargado con éxito")
            except Exception as e:
                print(f"❌ Error al cargar YOLOv8: {e}")
        
        self.lr_queue = None
        self.pro_queue = None
        self.face_queue = None
        self.depth_queue = None
        self.pro_depth_queue = None
        self.landmark_input_queue = None
        self.landmark_output_queue = None
        
        if not self.setup_oak_devices():
            raise RuntimeError("No fue posible asignar las cámaras OAK-D")
        
        self.load_selected_model()
    
    def load_extrinsic_calibration(self, calibration_path):
        """Load the published PRO-to-LR rigid transform."""
        path = Path(calibration_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            with path.open("r", encoding="utf-8") as handle:
                calibration = json.load(handle)
            rotation = np.asarray(calibration["R"], dtype=np.float64)
            translation = np.asarray(calibration["T"], dtype=np.float64).reshape(3)
            translation_unit = calibration.get("translation_unit")
            if translation_unit == "mm":
                translation = translation / 1000.0
            elif translation_unit != "m":
                raise ValueError("translation_unit must be explicitly 'm' or 'mm'")
            rotation = validate_proper_rotation(rotation)
            if translation.shape != (3,) or not np.isfinite(translation).all():
                raise ValueError("T must be a finite three-vector")
            self.R_lr_from_pro = rotation
            self.T_lr_from_pro_m = translation
            self.extrinsic_rmse_mm = calibration.get("rms_mm")
            if "K_lr" in calibration:
                self.K_lr = np.asarray(calibration["K_lr"], dtype=np.float64)
            print(f"Calibración extrínseca cargada: {path.name}")
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar la calibración extrínseca {path}: {exc}") from exc

    def validate_required_models(self):
        """Validate local model files and prevent implicit downloads."""
        configured = {
            "yolo_model": self.args.yolo_model,
            "gaze_model": self.args.gaze_model,
            "face_detector_model": self.args.face_detector_model,
            "landmark_model": self.args.landmark_model,
        }
        resolved = {}
        missing = []
        for key, value in configured.items():
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path = path.resolve()
            resolved[key] = path
            if not path.is_file():
                missing.append(f"{key}: {path}")
        if missing:
            details = "\n  - ".join(missing)
            raise FileNotFoundError(
                "Faltan modelos externos; no se descargarán automáticamente:\n  - "
                f"{details}\nConsulta models/README.md."
            )
        return resolved

    def setup_oak_devices(self):
        """Initialize the OAK-D LR and Pro devices."""
        print("🔍 Buscando dispositivos OAK...")
        device_infos = dai.Device.getAllAvailableDevices()
        
        if len(device_infos) < 2:
            print("❌ Se necesitan 2 dispositivos OAK-D (LR y Pro)")
            return False
        
        device_lr = None
        device_pro = None
        
        # USB enumeration order is not a semantic camera role. Requiring both
        # identifiers prevents applying PRO->LR geometry to swapped streams.
        if bool(self.args.lr_device_id) != bool(self.args.pro_device_id):
            raise ValueError("Debes indicar juntos --lr-device-id y --pro-device-id")
        if self.args.lr_device_id and self.args.pro_device_id:
            if self.args.lr_device_id == self.args.pro_device_id:
                raise ValueError("LR y Pro no pueden usar el mismo MXID")
            by_id = {str(info.getMxId()): info for info in device_infos}
            device_lr = by_id.get(self.args.lr_device_id)
            device_pro = by_id.get(self.args.pro_device_id)
            if device_lr is None or device_pro is None:
                available = ", ".join(sorted(by_id))
                raise RuntimeError(f"MXID solicitado no disponible. Detectados: {available}")
        elif self.args.allow_auto_device_order:
            print("ADVERTENCIA: asignando LR/Pro por orden USB; la geometría puede quedar invertida")
            device_lr, device_pro = device_infos[:2]
        else:
            available = ", ".join(str(info.getMxId()) for info in device_infos)
            raise RuntimeError(
                "La asignación automática de LR/Pro está desactivada. "
                f"Detectados: {available}. Usa --lr-device-id y --pro-device-id."
            )
        
        if not device_lr or not device_pro:
            print("❌ No se encontraron los dispositivos OAK-D específicos")
            return False
        
        # Road-facing OAK-D LR.
        self.device_lr = dai.Device(device_lr)
        lr_product = str(self.device_lr.getProductName() or self.device_lr.getDeviceName())
        if "PRO" in lr_product.upper():
            self.device_lr.close()
            raise RuntimeError(
                f"El MXID indicado como LR pertenece a {lr_product}; los roles están intercambiados"
            )
        lr_pipeline = dai.Pipeline()
        
        cam_lr = lr_pipeline.create(dai.node.ColorCamera)
        cam_lr.setPreviewSize(640, 360)
        # OAK-D LR uses AR0234 1200p color sensors; 1080p belongs to the Pro's
        # IMX378 and only appeared to work while the devices were swapped.
        cam_lr.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        # Match the 30 FPS RGB capture rate reported in the paper.
        cam_lr.setFps(30)
        cam_lr.setInterleaved(False)
        cam_lr.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        xout_lr = lr_pipeline.create(dai.node.XLinkOut)
        xout_lr.setStreamName("lr")
        cam_lr.preview.link(xout_lr.input)
        
        # ColorCamera nodes enable ISP scaling for the LR AR0234 stereo pair.
        mono_left = lr_pipeline.create(dai.node.ColorCamera)
        mono_right = lr_pipeline.create(dai.node.ColorCamera)
        stereo = lr_pipeline.create(dai.node.StereoDepth)
        
        mono_left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_left.setFps(30)
        # OV9782 sensors expose 1280x800. A 2/3 ISP scale produces an
        # unsupported 854 px width; 1/2 keeps the stereo input at 640x400.
        mono_left.setIspScale(1, 2)
        
        mono_right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_right.setFps(30)
        mono_right.setIspScale(1, 2)
        
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(640, 360)
        
        mono_left.isp.link(stereo.left)
        mono_right.isp.link(stereo.right)
        
        xout_depth = lr_pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        self.device_lr.startPipeline(lr_pipeline)
        try:
            calibration = self.device_lr.readCalibration()
            socket = getattr(dai.CameraBoardSocket, "CAM_A", dai.CameraBoardSocket.RGB)
            self.K_lr = np.asarray(
                calibration.getCameraIntrinsics(socket, self.frame_width, self.frame_height),
                dtype=np.float64,
            )
        except Exception as exc:
            print(f"Advertencia: se usarán intrínsecos del archivo de calibración: {exc}")
        self.lr_queue = self.device_lr.getOutputQueue("lr", 4, False)
        self.depth_queue = self.device_lr.getOutputQueue("depth", 4, False)
        
        # Driver-facing OAK-D Pro.
        self.device_pro = dai.Device(device_pro)
        pro_product = str(self.device_pro.getProductName() or self.device_pro.getDeviceName())
        if "PRO" not in pro_product.upper():
            self.device_pro.close()
            self.device_lr.close()
            raise RuntimeError(
                f"El MXID indicado como Pro pertenece a {pro_product}; los roles están intercambiados"
            )
        pro_pipeline = dai.Pipeline()
        
        cam_pro = pro_pipeline.create(dai.node.ColorCamera)
        cam_pro.setPreviewSize(640, 360)
        cam_pro.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_pro.setFps(30)
        cam_pro.setInterleaved(False)
        cam_pro.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Driver-facing RGB-D depth aligned with the RGB stream.
        pro_left = pro_pipeline.create(dai.node.MonoCamera)
        pro_right = pro_pipeline.create(dai.node.MonoCamera)
        pro_stereo = pro_pipeline.create(dai.node.StereoDepth)
        pro_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        pro_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        pro_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        pro_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        pro_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        pro_stereo.setLeftRightCheck(True)
        pro_stereo.setSubpixel(True)
        pro_stereo.setDepthAlign(getattr(dai.CameraBoardSocket, "CAM_A", dai.CameraBoardSocket.RGB))
        pro_stereo.setOutputSize(self.frame_width, self.frame_height)
        pro_left.out.link(pro_stereo.left)
        pro_right.out.link(pro_stereo.right)

        pro_depth_xout = pro_pipeline.create(dai.node.XLinkOut)
        pro_depth_xout.setStreamName("pro_depth")
        pro_stereo.depth.link(pro_depth_xout.input)
        
        # Face detection operates on a resized input.
        face_manip = pro_pipeline.create(dai.node.ImageManip)
        face_manip.initialConfig.setResize(300, 300)
        face_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        cam_pro.preview.link(face_manip.inputImage)
        
        face_det_nn = pro_pipeline.create(dai.node.MobileNetDetectionNetwork)
        face_det_nn.setBlobPath(str(self.asset_paths["face_detector_model"]))
        face_det_nn.setConfidenceThreshold(0.5)
        face_manip.out.link(face_det_nn.input)
        
        face_det_xout = pro_pipeline.create(dai.node.XLinkOut)
        face_det_xout.setStreamName("face_detections")
        face_det_nn.out.link(face_det_xout.input)

        # The host sends a 60x60 face crop to the 35-point landmark model.
        landmark_in = pro_pipeline.create(dai.node.XLinkIn)
        landmark_in.setStreamName("landmark_in")
        landmark_nn = pro_pipeline.create(dai.node.NeuralNetwork)
        landmark_nn.setBlobPath(str(self.asset_paths["landmark_model"]))
        landmark_in.out.link(landmark_nn.input)
        landmark_xout = pro_pipeline.create(dai.node.XLinkOut)
        landmark_xout.setStreamName("landmark_out")
        landmark_nn.out.link(landmark_xout.input)
        
        xout_pro = pro_pipeline.create(dai.node.XLinkOut)
        xout_pro.setStreamName("pro")
        cam_pro.preview.link(xout_pro.input)
        
        self.device_pro.startPipeline(pro_pipeline)
        try:
            pro_calibration = self.device_pro.readCalibration()
            pro_socket = getattr(dai.CameraBoardSocket, "CAM_A", dai.CameraBoardSocket.RGB)
            self.K_pro = np.asarray(
                pro_calibration.getCameraIntrinsics(pro_socket, self.frame_width, self.frame_height),
                dtype=np.float64,
            )
        except Exception as exc:
            print(f"Advertencia: se usarán intrínsecos aproximados para la cámara interior: {exc}")
        self.pro_queue = self.device_pro.getOutputQueue("pro", 4, False)
        self.face_queue = self.device_pro.getOutputQueue("face_detections", 4, False)
        self.pro_depth_queue = self.device_pro.getOutputQueue("pro_depth", 4, False)
        self.landmark_input_queue = self.device_pro.getInputQueue("landmark_in")
        self.landmark_output_queue = self.device_pro.getOutputQueue("landmark_out", 2, True)
        
        print(f"Cámaras verificadas: LR={lr_product}, conductor={pro_product}")
        return True
    
    def load_gaze_model(self, model_name):
        """Load the ResNet-101 model selected in the paper."""
        configs = {
            "resnet101_v2": {
                "path": str(self.asset_paths["gaze_model"]),
                "builder": get_gaze_model,
                "strip_prefix": False,
                "extract_key": None,
                # The model returns (pitch, yaw); the internal interface uses (yaw, pitch).
                "swap_axes": True,
                "invert_y": True    # Adapt the vertical sign to the image Y axis.
            }
        }
        
        if model_name == "openvino_fallback":
            print("🔄 Usando fallback 2D geométrico")
            return None
        
        config = configs.get(model_name)
        if not config:
            print(f"❌ Configuración no encontrada para {model_name}")
            return None
        
        try:
            model_path = Path(config["path"])
            if not model_path.exists():
                print(f"❌ Archivo no encontrado: {model_path}")
                return None
            
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = _extract_sd(checkpoint)
            
            model = config["builder"]()
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            print(f"✅ Modelo cargado: {model_name}")
            return model, config
            
        except Exception as e:
            print(f"❌ Error cargando {model_name}: {e}")
            return None, {}
    
    def load_selected_model(self):
        """Load the gaze model selected for the system."""
        self.current_model_name = self.model_names[self.model_index]
        result = self.load_gaze_model(self.current_model_name)
        
        if isinstance(result, tuple):
            self.current_model, self.current_model_config = result
        else:
            self.current_model = result
            self.current_model_config = {}
        
        self.current_session = []
        self.waiting_for_click = False
        
        print(f"🔄 Modelo actual: {self.current_model_name}")
        if self.current_model_config.get('swap_axes'):
            print("   ℹ️ Salida del modelo (pitch, yaw) normalizada a (yaw, pitch)")
        if self.current_model_config.get('invert_y'):
            print("   ⚠️ Eje Y invertido (arriba/abajo)")
        
        return self.current_model_name
    
    def predict_gaze(self, face_crop, model, swap_axes=False, invert_y=False):
        """Return ``(yaw, pitch)`` in radians from a normalized face crop."""
        if model is None:
            return 0.0, 0.0
        try:
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_tensor = transform(face_rgb).unsqueeze(0)
            with torch.no_grad():
                output = model(face_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            gaze = np.asarray(output.cpu().numpy()).flatten()
            if gaze.size < 2:
                return 0.0, 0.0
            val1, val2 = float(gaze[0]), float(gaze[1])
            # Canonical order used everywhere after this point: (yaw, pitch).
            yaw, pitch = (val2, val1) if swap_axes else (val1, val2)
            if invert_y:
                pitch = -pitch
            return yaw, pitch
        except Exception as exc:
            print(f"Error en predicción de mirada: {exc}")
            return 0.0, 0.0

    def align_face_with_landmarks(self, pro_frame, face_box):
        """Detect 35 landmarks and produce a 224x224 aligned face crop."""
        x1, y1, x2, y2 = face_box
        crop = pro_frame[y1:y2, x1:x2]
        if crop.size == 0 or self.landmark_input_queue is None:
            return None, None
        resized = cv2.resize(crop, (60, 60), interpolation=cv2.INTER_LINEAR)
        frame = dai.ImgFrame()
        frame.setType(dai.ImgFrame.Type.BGR888p)
        frame.setWidth(60)
        frame.setHeight(60)
        frame.setData(np.ascontiguousarray(resized.transpose(2, 0, 1)).flatten())
        self.landmark_input_queue.send(frame)
        output = self.landmark_output_queue.get()
        values = np.asarray(output.getFirstLayerFp16(), dtype=np.float32)
        if values.size != 70 or not np.all(np.isfinite(values)):
            return None, None
        points = values.reshape(35, 2)
        points[:, 0] = x1 + np.clip(points[:, 0], 0.0, 1.0) * (x2 - x1)
        points[:, 1] = y1 + np.clip(points[:, 1], 0.0, 1.0) * (y2 - y1)

        left_eye = points[[0, 1]].mean(axis=0)
        right_eye = points[[2, 3]].mean(axis=0)
        mouth = points[[8, 9]].mean(axis=0)
        try:
            transform = build_face_alignment_transform(left_eye, right_eye, mouth)
        except ValueError:
            return None, points
        aligned = cv2.warpAffine(
            pro_frame, transform, (224, 224),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned, points

    def estimate_face_origin_pro(self, face_box):
        """Estimate the 3D face center from driver-facing RGB-D depth."""
        if self.last_pro_depth_frame is None:
            return None
        x1, y1, x2, y2 = face_box
        margin_x = max(1, int((x2 - x1) * 0.30))
        margin_y = max(1, int((y2 - y1) * 0.30))
        roi = self.last_pro_depth_frame[
            max(0, y1 + margin_y):min(self.frame_height, y2 - margin_y),
            max(0, x1 + margin_x):min(self.frame_width, x2 - margin_x),
        ]
        valid = roi[(roi > 150) & (roi < 5000)]
        if valid.size < 10:
            return None
        z = float(np.median(valid)) / 1000.0
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        x = (u - self.K_pro[0, 2]) * z / self.K_pro[0, 0]
        y = (v - self.K_pro[1, 2]) * z / self.K_pro[1, 1]
        return np.array([x, y, z], dtype=np.float64)

    def project_gaze_to_lr(self, gaze_angles, depth_frame=None, origin_pro=None):
        """Transform model ray -> PRO ray -> LR ray and project it.

        Returns ``(None, direction_lr)`` for invalid geometry. It never invents
        a center point or clips an out-of-FOV ray onto an image border.
        """
        yaw, pitch = gaze_angles
        direction_model = model_angles_to_direction(yaw, pitch)
        if self.R_pro_from_model is None:
            self.last_projection_status = "missing-model-to-pro-calibration"
            return None, np.full(3, np.nan, dtype=np.float64)
        direction_pro = self.R_pro_from_model @ direction_model
        direction_lr = self.R_lr_from_pro @ direction_pro
        norm = np.linalg.norm(direction_lr)
        if norm < 1e-9 or direction_lr[2] <= 1e-6:
            self.last_projection_status = "ray-behind-lr-camera"
            return None, direction_lr
        direction_lr /= norm
        if origin_pro is None:
            origin_pro = np.zeros(3, dtype=np.float64)
        # Both terms are metres: RGB-D origin is converted above and T is normalized on load.
        origin_lr = self.R_lr_from_pro @ np.asarray(origin_pro, dtype=np.float64) + self.T_lr_from_pro_m

        target_depth = 10.0
        projected_point = None
        for _ in range(2):
            ray_scale = (target_depth - origin_lr[2]) / direction_lr[2]
            if ray_scale <= 0:
                self.last_projection_status = "intersection-behind-origin"
                return None, direction_lr
            point_lr = origin_lr + ray_scale * direction_lr
            if point_lr[2] <= 1e-6:
                self.last_projection_status = "intersection-behind-lr-camera"
                return None, direction_lr
            projected = self.K_lr @ point_lr
            u = int(round(projected[0] / projected[2]))
            v = int(round(projected[1] / projected[2]))
            if not (0 <= u < self.frame_width and 0 <= v < self.frame_height):
                self.last_projection_status = "ray-outside-lr-fov"
                return None, direction_lr
            projected_point = (u, v)
            if depth_frame is not None:
                x1, x2 = max(0, u - 5), min(self.frame_width, u + 6)
                y1, y2 = max(0, v - 5), min(self.frame_height, v + 6)
                valid = depth_frame[y1:y2, x1:x2]
                valid = valid[valid > 0]
                if valid.size:
                    target_depth = float(np.median(valid) / 1000.0)
        if projected_point is None:
            self.last_projection_status = "no-valid-intersection"
            return None, direction_lr
        self.last_projection_status = "ok"
        return projected_point, direction_lr

    def smooth_gaze_point(self, gaze_point):
        """Apply the causal EMA used by both the ADAS decision and the UI."""
        self.gaze_history.append(gaze_point)
        if len(self.gaze_history) < 2:
            return gaze_point
        alpha = 0.7
        prev_x, prev_y = self.gaze_history[-2]
        return (
            int(alpha * gaze_point[0] + (1.0 - alpha) * prev_x),
            int(alpha * gaze_point[1] + (1.0 - alpha) * prev_y),
        )

    def calculate_error(self, gaze_point, click_point):
        """Calculate Euclidean distance error in pixels."""
        dx = gaze_point[0] - click_point[0]
        dy = gaze_point[1] - click_point[1]
        return math.sqrt(dx*dx + dy*dy)

    def calculate_angular_error(self, gaze_point, click_point):
        """
        Calculate the angular gaze error between the prediction and GT point.

        Both points are converted to 3D rays using a pinhole camera model:
            r = [(x - cx) / fx, (y - cy) / fy, 1]

        Return the result in degrees using the active LR camera intrinsics.
        """
        fx = float(self.K_lr[0, 0])
        fy = float(self.K_lr[1, 1])
        cx = float(self.K_lr[0, 2])
        cy = float(self.K_lr[1, 2])

        x_pred, y_pred = gaze_point
        x_gt, y_gt = click_point

        ray_pred = np.array([
            (float(x_pred) - cx) / fx,
            (float(y_pred) - cy) / fy,
            1.0
        ], dtype=np.float64)

        ray_gt = np.array([
            (float(x_gt) - cx) / fx,
            (float(y_gt) - cy) / fy,
            1.0
        ], dtype=np.float64)

        norm_pred = np.linalg.norm(ray_pred)
        norm_gt = np.linalg.norm(ray_gt)
        if norm_pred < 1e-9 or norm_gt < 1e-9:
            return 0.0

        ray_pred /= norm_pred
        ray_gt /= norm_gt

        dot_product = float(np.clip(np.dot(ray_pred, ray_gt), -1.0, 1.0))
        angle_rad = math.acos(dot_product)
        return float(math.degrees(angle_rad))
    
    def calculate_nss(self, gaze_point, click_point):
        """Compute NSS on a Gaussian map with sigma = 22 px."""
        h, w = self.frame_height, self.frame_width
        sigma = 22.0

        yy, xx = np.mgrid[0:h, 0:w]
        gx, gy = gaze_point
        saliency_map = np.exp(
            -((xx - gx) ** 2 + (yy - gy) ** 2)
            / (2.0 * sigma ** 2)
        )

        std = saliency_map.std()
        if std < 1e-6:
            return 0.0
        saliency_norm = (saliency_map - saliency_map.mean()) / std

        cx = int(np.clip(click_point[0], 0, w - 1))
        cy = int(np.clip(click_point[1], 0, h - 1))
        return float(saliency_norm[cy, cx])
    
    def add_sample(self, gaze_point, click_point, model_name):
        """Add test sample to statistics."""
        error = self.calculate_error(gaze_point, click_point)
        angular_error = self.calculate_angular_error(gaze_point, click_point)
        nss = self.calculate_nss(gaze_point, click_point)
        
        sample = {
            'timestamp': time.time(),
            'model': model_name,
            'gaze_point': [int(gaze_point[0]), int(gaze_point[1])],
            'click_point': [int(click_point[0]), int(click_point[1])],
            'error_px': float(error),
            'angular_error_deg': float(angular_error),
            'nss': float(nss)
        }
        
        self.test_samples.append(sample)
        self.current_session.append(sample)
        
        stats = self.model_stats[model_name]
        stats['samples'] += 1
        stats['total_error'] += error
        stats['errors'].append(float(error))
        stats['angular_errors'].append(float(angular_error))
        stats['click_times'].append(time.time())
        stats['nss_scores'].append(float(nss))
        
        return error

    def save_lr_frame_for_yolo_dataset(self, frame=None):
        """Save an unlabeled LR frame for the YOLO validation dataset."""
        try:
            if frame is None:
                frame = self.last_frame
            if frame is None:
                print("⚠️ No hay frame LR disponible todavía para guardar.")
                return None
            if not self.args.save_dataset_frames:
                print("Guardado desactivado. Usa --save-dataset-frames para habilitarlo.")
                return
            save_dir = self.output_dir / "dataset_adas/images/raw"
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"lr_frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"📸 Frame LR guardado para etiquetar: {out_path}")
            return out_path
        except Exception as e:
            print(f"❌ Error guardando frame LR para dataset YOLO: {e}")
            return None
    
    def process_detections(self, boxes, cls_ids, masks_xy, names, lr_frame, gaze_point):
        """Compute distance, category, lane, gaze, and TTC for each detection."""
        current_time = time.time()
        
        # Reset counters for the current frame.
        self.current_counts = {
            'reliable_stereo': 0,
            'far_stereo': 0,
            'persona': 0,
            'vehiculo': 0,
            'camion_bus': 0,
            'bicicleta': 0,
            'motocicleta': 0,
            'lane_total': 0,
            'lane_unseen': 0,
            'lane_critical_ttc': 0
        }
        
        current_detections = []
        danger_in_lane = False
        frame_has_preliminary_risk = False
        frame_has_ttc_positive = False
        
        # Estimate depth and collect detections.
        for i, box in enumerate(boxes):
            class_id = int(cls_ids[i])
            obj_name = names[class_id].upper()
            
            # Map detector classes to the evaluated categories.
            category = None
            if obj_name == "PERSON":
                category = "persona"
            elif obj_name == "CAR":
                category = "vehiculo"
            elif obj_name in ["TRUCK", "BUS"]:
                category = "camion_bus"
            elif obj_name == "BICYCLE":
                category = "bicicleta"
            elif obj_name == "MOTORCYCLE":
                category = "motocicleta"
                
            if category is None:
                continue
                
            x1_b, y1_b, x2_b, y2_b = box
            cx = (x1_b + x2_b) / 2.0
            cy = (y1_b + y2_b) / 2.0
            
            # Estimate depth at the detection centroid.
            depth_m = None
            if self.last_depth_frame is not None:
                center_x = max(0, min(self.frame_width - 1, int(round(cx))))
                center_y = max(0, min(self.frame_height - 1, int(round(cy))))
                center_depth_mm = int(self.last_depth_frame[center_y, center_x])
                if center_depth_mm > 0:
                    depth_m = center_depth_mm / 1000.0
            
            # If no depth estimated, default to >30m (representing far or undetected by stereo)
            depth_val = depth_m if depth_m is not None else 999.0
            
            current_detections.append({
                'class_id': class_id,
                'category': category,
                'obj_name': obj_name,
                'box': box,
                'cx': cx,
                'cy': cy,
                'depth': depth_val,
                'x_m': ((cx - self.K_lr[0, 2]) * depth_val / self.K_lr[0, 0]) if depth_val < 900.0 else float('inf'),
                'y_m': ((cy - self.K_lr[1, 2]) * depth_val / self.K_lr[1, 1]) if depth_val < 900.0 else float('inf'),
                'mask_idx': i
            })
            if depth_val <= 30.0:
                frame_has_preliminary_risk = True

        if frame_has_preliminary_risk:
            self.risk_condition_frames += 1
            
        # Associate objects across frames for TTC estimation.
        new_tracked_objects = {}
        for det in current_detections:
            best_match_id = None
            best_distance = float('inf')
            for track_id, track in self.tracked_objects.items():
                if track['class_id'] == det['class_id']:
                    dist = math.sqrt((track['cx'] - det['cx'])**2 + (track['cy'] - det['cy'])**2)
                    if dist < 60 and dist < best_distance:
                        best_distance = dist
                        best_match_id = track_id
            
            ttc = float('inf')
            if best_match_id is not None:
                prev_track = self.tracked_objects[best_match_id]
                dt = current_time - prev_track['timestamp']
                depth_history = deque(prev_track.get('depth_history', []), maxlen=self.depth_history_length)
                if det['depth'] < 900.0:
                    depth_history.append(det['depth'])
                smoothed_depth = float(np.mean(depth_history)) if depth_history else det['depth']
                previous_smoothed_depth = prev_track.get('smoothed_depth', prev_track['depth'])
                if dt > 0.01 and previous_smoothed_depth < 900.0 and smoothed_depth < 900.0:
                    d_diff = previous_smoothed_depth - smoothed_depth
                    rel_speed = d_diff / dt
                    if rel_speed > 0.1:
                        ttc = smoothed_depth / rel_speed
                else:
                    ttc = prev_track.get('ttc', float('inf'))
                
                track_id_assigned = best_match_id
            else:
                track_id_assigned = self.next_track_id
                self.next_track_id += 1
                depth_history = deque(maxlen=self.depth_history_length)
                if det['depth'] < 900.0:
                    depth_history.append(det['depth'])
                smoothed_depth = det['depth']
                
            new_tracked_objects[track_id_assigned] = {
                'class_id': det['class_id'],
                'category': det['category'],
                'obj_name': det['obj_name'],
                'cx': det['cx'],
                'cy': det['cy'],
                'depth': det['depth'],
                'depth_history': list(depth_history),
                'smoothed_depth': smoothed_depth,
                'timestamp': current_time,
                'ttc': ttc,
                'box': det['box']
            }
            det['ttc'] = ttc
            det['track_id'] = track_id_assigned
            
        self.tracked_objects = new_tracked_objects
        
        # Apply lane, gaze, and warning conditions.
        lane_half_width_m = 1.75
        
        for det in current_detections:
            depth_val = det['depth']
            category = det['category']
            box = det['box']
            x1_b, y1_b, x2_b, y2_b = box
            
            # Count depth-range observations.
            if depth_val <= 30.0:
                self.current_counts['reliable_stereo'] += 1
            else:
                self.current_counts['far_stereo'] += 1
                
            # Count object categories.
            self.current_counts[category] += 1
            if self.vehicle_moving:
                self.total_detections_while_moving += 1
            
            # A 3.5 m corridor in LR camera coordinates. The object's projected
            # horizontal extent is used instead of a fixed image rectangle.
            if depth_val < 900.0:
                x_left_m = (x1_b - self.K_lr[0, 2]) * depth_val / self.K_lr[0, 0]
                x_right_m = (x2_b - self.K_lr[0, 2]) * depth_val / self.K_lr[0, 0]
                in_lane = x_right_m >= -lane_half_width_m and x_left_m <= lane_half_width_m
            else:
                in_lane = False
            
            # Test the bounding box before the more precise instance mask.
            gaze_in_box = (x1_b <= gaze_point[0] <= x2_b and y1_b <= gaze_point[1] <= y2_b)
            gaze_in_mask = False
            
            if gaze_in_box and masks_xy is not None:
                poly = masks_xy[det['mask_idx']].astype(np.int32)
                if len(poly) > 0:
                    inside = cv2.pointPolygonTest(poly, (float(gaze_point[0]), float(gaze_point[1])), False)
                    if inside >= 0:
                        gaze_in_mask = True
                        
            driver_saw_it = gaze_in_mask or (gaze_in_box and masks_xy is None)
            
            if in_lane:
                self.current_counts['lane_total'] += 1
                
                # Count unattended objects in the lane corridor.
                if not driver_saw_it:
                    self.current_counts['lane_unseen'] += 1
                    
                # Evaluate the critical TTC threshold.
                ttc = det['ttc']
                if (
                    self.vehicle_moving
                    and 0.0 < depth_val <= 30.0
                    and ttc <= self.ttc_threshold
                ):
                    frame_has_ttc_positive = True
                    self.current_counts['lane_critical_ttc'] += 1
                    self.ttc_positive_records += 1
                    # Warn only when the driver is not looking at the object.
                    if not driver_saw_it:
                        danger_in_lane = True
                        self.unattended_ttc_positive_records += 1
                        
            # Draw detection state using the warning color convention.
            if in_lane:
                if not driver_saw_it:
                    # Not seen and in lane -> Orange or Red depending on TTC
                    color = (0, 0, 255) if det['ttc'] <= self.ttc_threshold else (0, 165, 255)
                    thickness = 3
                else:
                    color = (0, 255, 0)
                    thickness = 2
            else:
                color = (150, 150, 150)
                thickness = 1
                
            cv2.rectangle(lr_frame, (int(x1_b), int(y1_b)), (int(x2_b), int(y2_b)), color, thickness)
            
            label = f"{det['obj_name']}"
            if depth_val < 900.0:
                label += f" {depth_val:.1f}m"
            else:
                label += " >30m"
                
            if det['ttc'] < 900.0 and det['ttc'] != float('inf'):
                label += f" TTC:{det['ttc']:.1f}s"
                
            cv2.putText(lr_frame, label, (int(x1_b), int(y1_b) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        
            # Highlight the attended instance mask.
            if driver_saw_it and masks_xy is not None:
                poly = masks_xy[det['mask_idx']].astype(np.int32)
                if len(poly) > 0:
                    obj_overlay = lr_frame.copy()
                    cv2.fillPoly(obj_overlay, [poly], (0, 255, 0))
                    cv2.addWeighted(obj_overlay, 0.4, lr_frame, 0.6, 0, lr_frame)
                    
        if frame_has_ttc_positive:
            self.ttc_positive_frames += 1
        if danger_in_lane:
            self.alarm_positive_frames += 1
        return lr_frame, danger_in_lane

    def _msg_timestamp_seconds(self, msg):
        """Return a DepthAI timestamp in seconds, falling back to the host clock."""
        try:
            ts = msg.getTimestamp()
            if hasattr(ts, 'total_seconds'):
                return ts.total_seconds()
            return float(ts)
        except Exception:
            return time.perf_counter()

    def update_camera_fps(self, camera_name, msg):
        """Compute effective camera FPS from DepthAI message timestamps."""
        wall_now = time.perf_counter()
        if self._camera_first_wall_time[camera_name] is None:
            self._camera_first_wall_time[camera_name] = wall_now
        self._camera_last_wall_time[camera_name] = wall_now
        self.camera_frame_count[camera_name] += 1

        ts_now = self._msg_timestamp_seconds(msg)
        ts_prev = self._camera_last_ts[camera_name]
        self._camera_last_ts[camera_name] = ts_now

        if ts_prev is None:
            return

        dt = ts_now - ts_prev
        if dt <= 0:
            return

        fps = 1.0 / dt
        self.camera_fps[camera_name] = fps
        self.camera_fps_history[camera_name].append(fps)
        self.camera_avg_fps[camera_name] = (
            sum(self.camera_fps_history[camera_name]) / len(self.camera_fps_history[camera_name])
        )

    def get_total_camera_fps(self, camera_name):
        """Return the full-session average FPS for a camera."""
        first_t = self._camera_first_wall_time.get(camera_name)
        last_t = self._camera_last_wall_time.get(camera_name)
        count = self.camera_frame_count.get(camera_name, 0)
        if first_t is None or last_t is None or last_t <= first_t or count <= 1:
            return 0.0
        return (count - 1) / (last_t - first_t)

    def update_fps(self):
        """Compute main-loop FPS and print it once per second."""
        now = time.perf_counter()
        dt = now - self._fps_last_time
        self._fps_last_time = now

        if dt > 0:
            self.current_fps = 1.0 / dt
            self.fps_history.append(self.current_fps)
            self.avg_fps = sum(self.fps_history) / len(self.fps_history)
            self.fps_frame_count += 1

        # Limit console output to one update per second.
        if now - self._fps_last_print >= 1.0:
            print(
                f"⚡ FPS programa: {self.avg_fps:.1f} | "
                f"LR: {self.camera_avg_fps['LR']:.1f} | "
                f"Pro: {self.camera_avg_fps['Pro']:.1f}"
            )
            self._fps_last_print = now

    def draw_ui(self, frame, gaze_point):
        """Draw minimal clean UI"""
        h, w = frame.shape[:2]

        # Main-loop and camera-stream FPS.
        fps_text = (
            f"FPS Prog:{getattr(self, 'avg_fps', 0.0):.1f} "
            f"LR:{self.camera_avg_fps.get('LR', 0.0):.1f} "
            f"Pro:{self.camera_avg_fps.get('Pro', 0.0):.1f}"
        )
        cv2.putText(frame, fps_text, (w - 300, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        
        if self.gaze_valid:
            # Gaussian heatmap centered on the gaze point.
            x1 = gaze_point[0] - self.hm_radius
            y1 = gaze_point[1] - self.hm_radius
            x2 = x1 + self.hm_size
            y2 = y1 + self.hm_size
            hx1, hy1 = 0, 0
            hx2, hy2 = self.hm_size, self.hm_size
            if x1 < 0:  hx1 = -x1;          x1 = 0
            if y1 < 0:  hy1 = -y1;          y1 = 0
            if x2 > w:  hx2 = self.hm_size - (x2 - w); x2 = w
            if y2 > h:  hy2 = self.hm_size - (y2 - h); y2 = h
            if x1 < x2 and y1 < y2:
                roi = frame[y1:y2, x1:x2]
                hm_roi = self.heatmap_color[hy1:hy2, hx1:hx2]
                alpha = self.heatmap_alpha[hy1:hy2, hx1:hx2]
                frame[y1:y2, x1:x2] = (
                    hm_roi * alpha + roi * (1.0 - alpha)
                ).astype(np.uint8)

            # Gaze-point crosshair.
            cv2.circle(frame, gaze_point, 2, (255, 255, 255), -1)
            cv2.line(frame, (gaze_point[0]-50, gaze_point[1]), (gaze_point[0]+50, gaze_point[1]), (0, 255, 0), 1)
            cv2.line(frame, (gaze_point[0], gaze_point[1]-50), (gaze_point[0], gaze_point[1]+50), (0, 255, 0), 1)
        else:
            cv2.putText(
                frame, f"RAYO INVALIDO: {self.last_projection_status}", (10, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1,
            )

        # Evaluate the lane in metric coordinates; a fixed pixel band would
        # be inconsistent with perspective.
        cv2.putText(frame, "Carril fisico: +/-1.75 m", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Model name in the lower-right corner.
        cv2.putText(frame, self.current_model_name, (w - 180, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        
        # Object intersected by the gaze ray.
        objeto = getattr(self, 'current_looking_at', 'Fondo / Calle')
        if objeto not in ('Fondo', 'Fondo / Calle'):
            cv2.putText(frame, objeto, (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        
        # Depth at the gaze point.
        depth = getattr(self, 'current_depth_m', 0.0)
        if self.gaze_valid and depth > 0:
            cv2.putText(frame, f"{depth:.1f}m", (gaze_point[0] + 12, gaze_point[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Semi-transparent status panel.
        panel_w = 220
        panel_h = 175
        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (5, 5), (5 + panel_w, 5 + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(panel_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (5, 5), (5 + panel_w, 5 + panel_h), (80, 80, 80), 1)
        
        # Vehicle motion state.
        v_state = "MOVIMIENTO" if self.vehicle_moving else "DETENIDO"
        v_color = (0, 255, 0) if self.vehicle_moving else (0, 0, 255)
        cv2.putText(frame, f"Vehiculo: {v_state}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, v_color, 1)
        cv2.putText(frame, f"Magnitud: {self.motion_magnitude:.2f}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        cv2.line(frame, (10, 42), (panel_w, 42), (100, 100, 100), 1)
        
        # Stereo-depth ranges.
        cv2.putText(frame, f"Estereo Fiable (<=30m): {self.current_counts.get('reliable_stereo', 0)}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
        cv2.putText(frame, f"Estereo Lejano (>30m): {self.current_counts.get('far_stereo', 0)}", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1)
        
        # Object categories.
        cat_text = f"P:{self.current_counts.get('persona', 0)} | V:{self.current_counts.get('vehiculo', 0)} | C:{self.current_counts.get('camion_bus', 0)} | B:{self.current_counts.get('bicicleta', 0)} | M:{self.current_counts.get('motocicleta', 0)}"
        cv2.putText(frame, "Categorias:", (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, cat_text, (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        cv2.line(frame, (10, 104), (panel_w, 104), (100, 100, 100), 1)
        
        # Lane statistics.
        cv2.putText(frame, f"Total en carril: {self.current_counts.get('lane_total', 0)}", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(frame, f"No vistos en carril: {self.current_counts.get('lane_unseen', 0)}", (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1)
        cv2.putText(frame, f"TTC <= 2s en carril: {self.current_counts.get('lane_critical_ttc', 0)}", (12, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
        
        # Warning threshold.
        cv2.putText(frame, f"Alarma TTC Activa: {'SI' if self.alarm_active else 'NO'}", (12, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255) if self.alarm_active else (150, 150, 150), 1)
        
        # Flash a red border while a warning is active.
        if getattr(self, 'alarm_active', False):
            if int(time.time() * 5) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        
        return frame
    
    def generate_report(self):
        """Generate the validation and runtime report."""
        has_gaze_samples = bool(self.test_samples)
        has_alarm_data = self.total_frames_processed > 0
        if not has_gaze_samples and not has_alarm_data:
            print("❌ No hay datos para generar reporte (sin muestras de mirada ni eventos de alarma)")
            return
        if not has_gaze_samples:
            print("ℹ️  Sin muestras manuales de mirada — generando reporte solo con métricas de alarma")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(self.test_samples),
            'models': {},
            'raw_samples': self.test_samples,
            'method_configuration': {
                'extrinsics_file': str(self.args.extrinsics),
                'extrinsic_rmse_mm': self.extrinsic_rmse_mm,
                'R_lr_from_pro': self.R_lr_from_pro.tolist(),
                'T_lr_from_pro_m': self.T_lr_from_pro_m.tolist(),
                'K_lr_active': self.K_lr.tolist(),
                'K_pro_active': self.K_pro.tolist(),
                'yolo_model': self.args.yolo_model,
                'facial_landmark_model': self.args.landmark_model,
                'facial_landmarks_count': 35,
                'face_alignment_size': [224, 224],
                'driver_rgbd_depth_used_for_gaze_origin': True,
                'face_candidate_frames': self.face_candidate_frames,
                'landmark_valid_frames': self.landmark_valid_frames,
                'rgbd_face_depth_valid_frames': self.rgbd_face_depth_valid_frames,
                'yolo_confidence': 0.5,
                'max_depth_m': 30.0,
                'lane_width_m': 3.5,
                'motion_threshold_px': 1.5,
                'shi_tomasi_max_corners': 400,
                'lucas_kanade_window': [11, 11],
                'track_centroid_threshold_px': 60,
                'depth_smoothing_window': self.depth_history_length,
                'minimum_approach_speed_mps': 0.1,
                'ttc_threshold_s': self.ttc_threshold,
            },
        }
        
        for model_name in self.model_names:
            stats = self.model_stats[model_name]
            if stats['samples'] > 0:
                errors = np.array(stats['errors'], dtype=np.float64)
                angular_errors = np.array(stats['angular_errors'], dtype=np.float64)
                nss_scores = np.array(stats['nss_scores'], dtype=np.float64)
                report['models'][model_name] = {
                    'samples': int(stats['samples']),
                    'mean_error_px': float(np.mean(errors)),
                    'std_error_px': float(np.std(errors)),
                    'min_error_px': float(np.min(errors)),
                    'max_error_px': float(np.max(errors)),
                    'median_error_px': float(np.median(errors)),
                    'p95_error_px': float(np.percentile(errors, 95)),
                    'p99_error_px': float(np.percentile(errors, 99)),
                    'mean_angular_error_deg': float(np.mean(angular_errors)),
                    'std_angular_error_deg': float(np.std(angular_errors)),
                    'min_angular_error_deg': float(np.min(angular_errors)),
                    'max_angular_error_deg': float(np.max(angular_errors)),
                    'median_angular_error_deg': float(np.median(angular_errors)),
                    'p95_angular_error_deg': float(np.percentile(angular_errors, 95)),
                    'p99_angular_error_deg': float(np.percentile(angular_errors, 99)),
                    'mean_nss': float(np.mean(nss_scores)),
                    'std_nss': float(np.std(nss_scores)),
                    'pct_good_nss': float(np.mean(nss_scores > 1.0) * 100)
                }
        
        alarm_metrics = {
            'total_frames_processed': self.total_frames_processed,
            'risk_condition_frames': self.risk_condition_frames,
            'alarm_positive_frames': self.alarm_positive_frames,
            'audible_warning_count': self.alarm_trigger_count,
            'risk_condition_rate_pct': round(self.risk_condition_frames / max(self.total_frames_processed, 1) * 100, 2),
            'operational_activation_ratio_pct': round(self.alarm_positive_frames / max(self.risk_condition_frames, 1) * 100, 2),
        }
        if self.pipeline_latencies:
            alarm_metrics['latency_ms_mean']   = round(float(np.mean(self.pipeline_latencies)), 2)
            alarm_metrics['latency_ms_min']    = round(float(np.min(self.pipeline_latencies)), 2)
            alarm_metrics['latency_ms_max']    = round(float(np.max(self.pipeline_latencies)), 2)
            alarm_metrics['latency_ms_p95']    = round(float(np.percentile(self.pipeline_latencies, 95)), 2)
        report['alarm_metrics'] = alarm_metrics
        report['alarm_events']  = self.alarm_events
        
        adas_metrics = {
            'moving_frames_count': self.moving_frames_count,
            'total_detections_while_moving': self.total_detections_while_moving,
            'ttc_positive_records': self.ttc_positive_records,
            'unattended_ttc_positive_records': self.unattended_ttc_positive_records,
            'ttc_positive_frames': self.ttc_positive_frames,
            'ttc_positive_frame_rate_pct': round(self.ttc_positive_frames / max(self.moving_frames_count, 1) * 100, 2),
        }
        report['adas_metrics'] = adas_metrics

        # FPS values are saved when the user presses 'r' or exits.
        fps_metrics = {
            'program_fps_recent_avg': round(float(getattr(self, 'avg_fps', 0.0)), 2),
            'camera_lr_fps_recent_avg': round(float(self.camera_avg_fps.get('LR', 0.0)), 2),
            'camera_pro_fps_recent_avg': round(float(self.camera_avg_fps.get('Pro', 0.0)), 2),
            'camera_lr_fps_total_session': round(float(self.get_total_camera_fps('LR')), 2),
            'camera_pro_fps_total_session': round(float(self.get_total_camera_fps('Pro')), 2),
            'program_frames_processed_for_fps': int(getattr(self, 'fps_frame_count', 0)),
            'camera_lr_frames_received': int(self.camera_frame_count.get('LR', 0)),
            'camera_pro_frames_received': int(self.camera_frame_count.get('Pro', 0))
        }
        report['fps_metrics'] = fps_metrics
        
        if not self.args.save_reports:
            print("Reporte calculado en memoria; no se guardó (--save-reports para habilitarlo).")
            return report
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / f"adas_gaze_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*60)
        print("📊 INFORME DE VALIDACIÓN GAZE-AWARE ADAS")
        print("="*60)
        
        if report['models']:
            sorted_models = sorted(
                [(name, data['mean_error_px']) for name, data in report['models'].items()],
                key=lambda x: x[1]
            )
            
            for i, (model_name, mean_error) in enumerate(sorted_models, 1):
                data = report['models'][model_name]
                print(f"{i}. {model_name}")
                print(f"   Muestras: {data['samples']}")
                print(f"   Error medio: {data['mean_error_px']:.2f} ± {data['std_error_px']:.2f}px")
                print(f"   Rango px: {data['min_error_px']:.1f} - {data['max_error_px']:.1f}px")
                print(f"   P95 px: {data['p95_error_px']:.1f}px")
                print(f"   Error angular medio: {data['mean_angular_error_deg']:.2f} ± {data['std_angular_error_deg']:.2f}°")
                print(f"   Error angular mediano: {data['median_angular_error_deg']:.2f}°")
                print(f"   Error angular P95: {data['p95_angular_error_deg']:.2f}°")
                print(f"   NSS medio: {data['mean_nss']:.3f} ± {data['std_nss']:.3f}")
                print(f"   % muestras NSS > 1.0: {data['pct_good_nss']:.1f}%")
                print()
        else:
            print("   (No se registraron muestras manuales de mirada en esta sesión)")
        
        print("="*60)
        print("🔔 MÉTRICAS DE EFICIENCIA DE LA ALARMA")
        print("="*60)
        print(f"   Frames analizados: {alarm_metrics['total_frames_processed']}")
        print(f"   Frames con riesgo preliminar: {alarm_metrics['risk_condition_frames']}")
        print(f"   Frames con alarma positiva: {alarm_metrics['alarm_positive_frames']}")
        print(f"   Pitidos emitidos: {alarm_metrics['audible_warning_count']}")
        print(f"   Razón de activación operacional: {alarm_metrics['operational_activation_ratio_pct']:.1f}%")
        if self.pipeline_latencies:
            print(f"   Latencia media (captura→alarma): {alarm_metrics['latency_ms_mean']:.1f} ms")
            print(f"   Latencia mínima: {alarm_metrics['latency_ms_min']:.1f} ms")
            print(f"   Latencia máxima: {alarm_metrics['latency_ms_max']:.1f} ms")
            print(f"   Latencia P95: {alarm_metrics['latency_ms_p95']:.1f} ms")
        print()
        
        print("="*60)
        print("🚗 MÉTRICAS ADAS (CON VEHÍCULO EN MOVIMIENTO)")
        print("="*60)
        print(f"   Frames en movimiento (flujo optico + RANSAC): {adas_metrics['moving_frames_count']}")
        print(f"   Detecciones totales en movimiento: {adas_metrics['total_detections_while_moving']}")
        print(f"   Registros TTC <= 2s: {adas_metrics['ttc_positive_records']}")
        print(f"   Registros TTC <= 2s no atendidos: {adas_metrics['unattended_ttc_positive_records']}")
        print(f"   Frames TTC <= 2s: {adas_metrics['ttc_positive_frames']}")
        print()

        print("="*60)
        print("⚡ MÉTRICAS DE FPS")
        print("="*60)
        print(f"   FPS programa, promedio reciente: {fps_metrics['program_fps_recent_avg']:.2f}")
        print(f"   FPS cámara LR, promedio reciente: {fps_metrics['camera_lr_fps_recent_avg']:.2f}")
        print(f"   FPS cámara Pro, promedio reciente: {fps_metrics['camera_pro_fps_recent_avg']:.2f}")
        print(f"   FPS cámara LR, sesión completa: {fps_metrics['camera_lr_fps_total_session']:.2f}")
        print(f"   FPS cámara Pro, sesión completa: {fps_metrics['camera_pro_fps_total_session']:.2f}")
        print()
        
        print(f"📄 Reporte guardado en: {report_path}")
    
    def run(self):
        """Run the real-time ADAS processing loop."""
        print("🚀 Iniciando Gaze-Aware Stereo Vision ADAS...")
        print("📋 Controles:")
        print("   ESPACIO - Capturar punto de mirada")
        print("   Click - Indicar dónde estabas mirando")
        print("   'p' - Pausar/Reanudar (pausa la captura de frames)")
        print("   'r' - Generar reporte")
        print("   'q' - Salir")
        
        cv2.namedWindow("Gaze-Aware Stereo Vision ADAS")
        cv2.setMouseCallback("Gaze-Aware Stereo Vision ADAS", self.mouse_callback)
        
        video_writer = None
        video_path = None
        is_recording = self.args.record_video
        if is_recording:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            video_path = self.output_dir / f"grabacion_gaze_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 360))
        frame_count = 0
        start_time = time.time()
        print(f"Grabación: {'activada' if is_recording else 'desactivada'}")
        
        try:
            while True:
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    if video_writer is None:
                        print("Grabación desactivada. Reinicia con --record-video para habilitarla.")
                        continue
                    is_recording = not is_recording
                    print(f"{'🎥 Grabando' if is_recording else '⏸ Grabacion pausada'}")
                elif key == ord('p'):
                    self.is_paused = not self.is_paused
                    print(f"{'PAUSED' if self.is_paused else 'RESUMED'}")
                elif key == ord(' '):
                    self.waiting_for_click = True
                    self.click_point = None
                    print("🖱️ Click where you were looking...")
                elif key == ord('r'):
                    self.generate_report()
                elif key == 27:  # ESC
                    if self.waiting_for_click:
                        self.waiting_for_click = False
                        self.click_point = None
                        print("❌ Captura cancelada")
                    else:
                        self.click_point = None
                
                # While paused, continue displaying the last frame.
                if self.is_paused:
                    if self.last_frame is not None:
                        paused_frame = self.draw_ui(self.last_frame.copy(), self.last_gaze_point)
                        h, w = paused_frame.shape[:2]
                        cv2.putText(paused_frame, "PAUSED - Press 'p' to continue", (w//2 - 150, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imshow("Gaze-Aware Stereo Vision ADAS", paused_frame)
                    continue
                
                if self.lr_queue is None or self.pro_queue is None:
                    print("❌ Cámaras no inicializadas. Conecta los dispositivos OAK-D y reinicia.")
                    break
                
                lr_frame = None
                pro_frame = None
                face_detections = None
                
                # T0: driver-frame capture.
                t0_capture = time.perf_counter()

                lr_data = self.lr_queue.get()
                self.update_camera_fps('LR', lr_data)
                lr_frame = lr_data.getCvFrame()
                
                # Optical Flow + RANSAC vehicle motion detection
                gray = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2GRAY)
                if self.prev_gray is not None:
                    if self.prev_pts is None or len(self.prev_pts) < 30:
                        self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
                    
                    if self.prev_pts is not None and len(self.prev_pts) >= 4:
                        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)
                        if curr_pts is not None:
                            good_new = curr_pts[status == 1]
                            good_old = self.prev_pts[status == 1]
                            
                            if len(good_old) >= 4 and len(good_new) >= 4:
                                transform_matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                                if transform_matrix is not None:
                                    dx = transform_matrix[0, 2]
                                    dy = transform_matrix[1, 2]
                                    self.motion_magnitude = float(np.sqrt(dx**2 + dy**2))
                                    self.vehicle_moving = self.motion_magnitude > 1.5
                                else:
                                    self.vehicle_moving = False
                                    self.motion_magnitude = 0.0
                            else:
                                self.vehicle_moving = False
                                self.motion_magnitude = 0.0
                            
                            if len(good_new) > 0:
                                self.prev_pts = good_new.reshape(-1, 1, 2)
                            else:
                                self.prev_pts = None
                        else:
                            self.vehicle_moving = False
                            self.motion_magnitude = 0.0
                            self.prev_pts = None
                    else:
                        self.vehicle_moving = False
                        self.motion_magnitude = 0.0
                else:
                    self.vehicle_moving = False
                    self.motion_magnitude = 0.0
                
                self.prev_gray = gray.copy()
                if self.vehicle_moving:
                    self.moving_frames_count += 1
                
                depth_data = self.depth_queue.tryGet()
                if depth_data:
                    self.last_depth_frame = depth_data.getFrame()

                pro_data = self.pro_queue.get()
                self.update_camera_fps('Pro', pro_data)
                pro_frame = pro_data.getCvFrame()
                pro_depth_data = self.pro_depth_queue.tryGet() if self.pro_depth_queue else None
                if pro_depth_data is not None:
                    self.last_pro_depth_frame = pro_depth_data.getFrame()

                face_data = self.face_queue.tryGet()
                if face_data:
                    face_detections = face_data.detections
                
                # Process the face and predict gaze.
                gaze_point = self.last_gaze_point
                self.gaze_valid = False
                self.last_projection_status = "no-face-detection"
                
                if face_detections and len(face_detections) > 0:
                    self.face_candidate_frames += 1
                    face = face_detections[0]
                    # Detections are on 300x300 image, scale to 640x360
                    x1 = int(face.xmin * 300 * 640 / 300)
                    y1 = int(face.ymin * 300 * 360 / 300)
                    x2 = int(face.xmax * 300 * 640 / 300)
                    y2 = int(face.ymax * 300 * 360 / 300)
                    
                    if 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360:
                        face_box = (x1, y1, x2, y2)
                        face_crop, facial_landmarks = self.align_face_with_landmarks(pro_frame, face_box)
                        self.face_origin_pro = self.estimate_face_origin_pro(face_box)
                        if face_crop is not None:
                            self.landmark_valid_frames += 1
                            face_preview = cv2.flip(face_crop, 1)
                            cv2.putText(
                                face_preview, "Landmarks OK", (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                            )
                            cv2.imshow("NN Input (Face + Vector)", face_preview)
                            self.last_projection_status = "waiting-face-depth"
                        else:
                            self.last_projection_status = "invalid-face-landmarks"
                        if self.face_origin_pro is not None:
                            self.rgbd_face_depth_valid_frames += 1
                        elif face_crop is not None:
                            self.last_projection_status = "invalid-face-depth"
                        if face_crop is not None and self.face_origin_pro is not None:
                            # Normalize the model output to (yaw, pitch) and image-axis sign.
                            swap_axes = self.current_model_config.get('swap_axes', False)
                            invert_y = self.current_model_config.get('invert_y', False)
                            # T1: gaze-inference start.
                            t1_gaze_start = time.perf_counter()
                            gaze_angles = self.predict_gaze(face_crop, self.current_model, swap_axes=swap_axes, invert_y=invert_y)
                            t2_gaze_end = time.perf_counter()
                            projected_gaze_point, gaze_direction_lr = self.project_gaze_to_lr(
                                gaze_angles, self.last_depth_frame, self.face_origin_pro
                            )
                            self.gaze_valid = projected_gaze_point is not None
                            if self.gaze_valid:
                                # ADAS, UI and metrics use the same filtered gaze.
                                gaze_point = self.smooth_gaze_point(projected_gaze_point)
                            
                            # Run YOLOv8 segmentation and gaze-to-object mapping.
                            self.current_looking_at = (
                                "Fondo / Calle" if self.gaze_valid else "Rayo inválido"
                            )
                            danger_in_lane = False
                            if self.yolo_model is not None and self.gaze_valid:
                                results = self.yolo_model(lr_frame, conf=0.5, verbose=False)
                                if len(results) > 0:
                                    # Render masks without labels or boxes.
                                    yolo_annotated = results[0].plot(labels=False, boxes=False)
                                    lr_frame = yolo_annotated
                                    
                                    # Check whether the gaze point intersects an instance mask.
                                    if results[0].boxes is not None and results[0].masks is not None:
                                        boxes = results[0].boxes.xyxy.cpu().numpy()
                                        cls_ids = results[0].boxes.cls.cpu().numpy()
                                        masks_xy = results[0].masks.xy
                                        names = self.yolo_model.names
                                        
                                        # Traverse in reverse display order to resolve overlapping masks.
                                        for i, box in reversed(list(enumerate(boxes))):
                                            x1, y1, x2, y2 = box
                                            if x1 <= gaze_point[0] <= x2 and y1 <= gaze_point[1] <= y2:
                                                # Use the instance polygon for the precise intersection test.
                                                poly = masks_xy[i].astype(np.int32)
                                                if len(poly) > 0:
                                                    inside = cv2.pointPolygonTest(poly, (float(gaze_point[0]), float(gaze_point[1])), False)
                                                    if inside >= 0:
                                                        self.current_looking_at = names[int(cls_ids[i])].upper()
                                                        break
                                        
                                        # Update detection, lane, gaze, motion, and TTC state.
                                        lr_frame, danger_in_lane = self.process_detections(
                                            boxes, cls_ids, masks_xy, names, lr_frame, gaze_point
                                        )
                                    else:
                                        self.current_counts = {k: 0 for k in self.current_counts}
                                else:
                                    self.current_counts = {k: 0 for k in self.current_counts}
                            else:
                                self.current_counts = {k: 0 for k in self.current_counts}

                            # Emit an audible warning after the cooldown expires.
                            self.total_frames_processed += 1
                            if danger_in_lane:
                                self.frames_with_danger += 1
                                self.alarm_active = True
                                current_time = time.time()
                                t3_alarm = time.perf_counter()
                                latency_ms = None
                                if self.previous_frame_input_time is not None:
                                    latency_ms = (t3_alarm - self.previous_frame_input_time) * 1000
                                    self.pipeline_latencies.append(latency_ms)
                                gaze_latency_ms = (t2_gaze_end - t1_gaze_start) * 1000
                                should_beep = current_time - self.last_alarm_time > self.alarm_cooldown
                                self.alarm_events.append({
                                    'timestamp': time.time(),
                                    'latency_total_ms': round(latency_ms, 2) if latency_ms is not None else None,
                                    'latency_gaze_ms': round(gaze_latency_ms, 2),
                                    'object_missed': 'CRITICAL_LANE_OBJ',
                                    'driver_looking_at': self.current_looking_at,
                                    'beep_emitted': should_beep,
                                })
                                if should_beep:
                                    # T3: warning activation.
                                    t3_alarm = time.perf_counter()
                                    if winsound is not None:
                                        winsound.Beep(1000, 200)
                                    self.last_alarm_time = current_time
                                    self.alarm_trigger_count += 1
                                    
                                    # Compute total pipeline latency in milliseconds.
                                    latency_ms = (t3_alarm - self.previous_frame_input_time) * 1000 if self.previous_frame_input_time is not None else 0.0
                                    gaze_latency_ms = (t2_gaze_end - t1_gaze_start) * 1000
                                    
                                    print(f"🔔 ALARMA TTC | Peligro no visto en carril! | Conductor mira: {self.current_looking_at} | Latencia total: {latency_ms:.1f}ms")
                            else:
                                self.alarm_active = False
                            
                            # Preserve the validated debug orientation for the gaze overlay.
                            face_debug = cv2.flip(face_crop, 1)
                            h_face, w_face = face_debug.shape[:2]
                            
                            # Draw the gaze vector in normalized face coordinates.
                            # Preserve the model's horizontal-axis convention.
                            gaze_x_face = int(w_face / 2 - math.sin(gaze_angles[0]) * w_face / 2)
                            gaze_y_face = int(h_face / 2 - math.sin(gaze_angles[1]) * h_face / 2)
                            
                            center_x, center_y = w_face // 2, h_face // 2
                            
                            cv2.line(face_debug, (center_x, center_y), (gaze_x_face, gaze_y_face), (0, 255, 0), 2)
                            cv2.circle(face_debug, (gaze_x_face, gaze_y_face), 5, (0, 0, 255), -1)
                            cv2.circle(face_debug, (center_x, center_y), 3, (255, 0, 0), -1)
                            
                            cv2.putText(face_debug, f"Input: {face_crop.shape[1]}x{face_crop.shape[0]}", (5, 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            cv2.putText(face_debug, f"Yaw/Pitch: ({math.degrees(gaze_angles[0]):.1f}, {math.degrees(gaze_angles[1]):.1f}) deg", (5, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            
                            cv2.imshow("NN Input (Face + Vector)", face_debug)
                            self.last_face_debug = face_debug.copy()
                    else:
                        self.last_projection_status = "invalid-face-box"
                            

                self.last_gaze_point = gaze_point
                if lr_frame is not None:
                    self.last_frame = lr_frame.copy()
                
                # Update measured FPS before rendering the UI.
                self.update_fps()

                display_frame = self.draw_ui(lr_frame.copy(), gaze_point)
                
                cv2.imshow("Gaze-Aware Stereo Vision ADAS", display_frame)
                # Keep the complete driver-facing stream visible even when no
                # valid face/depth pair is available for gaze inference.
                pro_display = pro_frame.copy()
                cv2.putText(pro_display, "OAK-D Pro - Conductor", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("OAK-D Pro - Conductor", pro_display)
                self.previous_frame_input_time = t0_capture

                # Write the processed frame to the video stream.
                if is_recording and video_writer is not None:
                    video_writer.write(display_frame)
                    frame_count += 1
                
                if key == 27 and self.waiting_for_click:
                    self.waiting_for_click = False
                    self.click_point = None
                    print("❌ Captura cancelada")
        
        except KeyboardInterrupt:
            print("\n🛑 Programa interrumpido")
        
        finally:
            if self.device_lr:
                self.device_lr.close()
            if self.device_pro:
                self.device_pro.close()
            cv2.destroyAllWindows()
            
            # Finalize video output.
            if video_writer is not None:
                video_writer.release()
            elapsed_time = time.time() - start_time
            actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            processing_fps = self.fps_frame_count / elapsed_time if elapsed_time > 0 else 0
            video_duration = frame_count / 30.0  # Duration at the configured 30 FPS.
            if video_writer is not None:
                print(f"🎥 Video guardado en: {video_path}")
                print(f"📊 Estadísticas de grabación:")
                print(f"   Frames grabados: {frame_count}")
            print(f"   Tiempo real: {elapsed_time:.2f}s")
            if video_writer is not None:
                print(f"   FPS real grabado: {actual_fps:.2f}")
            print(f"   FPS real del programa: {processing_fps:.2f}")
            print(f"   FPS efectivo cámara LR: {self.get_total_camera_fps('LR'):.2f}")
            print(f"   FPS efectivo cámara Pro: {self.get_total_camera_fps('Pro'):.2f}")
            if video_writer is not None:
                print(f"   Duración del video: {video_duration:.2f}s")
            
            if self.args.save_reports:
                self.generate_report()
    
    def generate_heatmap_capture(self, frame, pred_point, click_point):
        """Generate a capture image with highlighted prediction and reference heatmaps."""
        h, w = frame.shape[:2]
        
        Y, X = np.mgrid[0:h, 0:w]
        sigma = 55.0
        
        # Prediction and reference Gaussians.
        gauss_pred = np.exp(-((X - pred_point[0])**2 + (Y - pred_point[1])**2) / (2 * sigma**2))
        gauss_click = np.exp(-((X - click_point[0])**2 + (Y - click_point[1])**2) / (2 * sigma**2))
        
        # Combine both heatmaps.
        saliency = np.maximum(gauss_pred, gauss_click)
        
        # Prefer TURBO when available and fall back to JET.
        cmap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
        
        saliency_8u = (saliency * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(saliency_8u, cmap)
        
        # Blend a subdued background with bright heatmap peaks.
        alpha_map = saliency * 0.7 + 0.3
        alpha_map = np.expand_dims(alpha_map, axis=2)
        
        # Darken the background before applying the heatmap.
        bg_tinted = frame.copy()
        bg_tinted = cv2.addWeighted(bg_tinted, 0.5, np.zeros_like(bg_tinted), 0.5, 0)
        
        overlay = (heatmap_color * alpha_map + bg_tinted * (1.0 - alpha_map)).astype(np.uint8)
        
        # Draw prediction and reference markers.
        cv2.circle(overlay, pred_point, 5, (255, 255, 255), -1)
        cv2.putText(overlay, "Prediccion", (pred_point[0] + 10, pred_point[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        cv2.circle(overlay, click_point, 5, (0, 255, 255), -1)
        cv2.putText(overlay, "Click GT", (click_point[0] + 10, click_point[1] + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
        # Connect the prediction and reference points.
        cv2.line(overlay, pred_point, click_point, (0, 255, 0), 2)
        
        error_dist = self.calculate_error(pred_point, click_point)
        angular_error = self.calculate_angular_error(pred_point, click_point)
        cv2.putText(overlay, f"Error: {error_dist:.1f}px | {angular_error:.2f} deg", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                   
        return overlay

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for ground truth"""
        if event == cv2.EVENT_LBUTTONDOWN and self.waiting_for_click:
            if not self.gaze_valid:
                print(f"Muestra rechazada: rayo inválido ({self.last_projection_status})")
                self.waiting_for_click = False
                return
            self.click_point = (x, y)
            self.error_distance = self.calculate_error(self.last_gaze_point, self.click_point)
            angular_error = self.calculate_angular_error(self.last_gaze_point, self.click_point)
            nss = self.calculate_nss(self.last_gaze_point, self.click_point)
            
            self.add_sample(self.last_gaze_point, self.click_point, self.current_model_name)
            
            # Captures may contain faces and are disabled by default.
            if not self.args.save_captures:
                print(f"Muestra | Error: {self.error_distance:.1f}px | Angular: {angular_error:.2f}° | NSS: {nss:.3f} (sin guardar)")
                self.waiting_for_click = False
                return
            try:
                import os
                save_dir = str(self.output_dir / "capturas_testing")
                os.makedirs(save_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                # Save the heatmap comparison.
                if hasattr(self, 'last_frame') and self.last_frame is not None:
                    frame_to_save = self.generate_heatmap_capture(self.last_frame.copy(), self.last_gaze_point, self.click_point)
                    cv2.imwrite(f"{save_dir}/{ts}_escena_{self.current_model_name}.jpg", frame_to_save)
                
                # Save the face crop with its gaze vector.
                if hasattr(self, 'last_face_debug') and self.last_face_debug is not None:
                    cv2.imwrite(f"{save_dir}/{ts}_sujeto_{self.current_model_name}.jpg", self.last_face_debug)
                    
                print(f"✅ Muestra | Error: {self.error_distance:.1f}px | Angular: {angular_error:.2f}° | NSS: {nss:.3f} {'🟢' if nss > 1.0 else '🔴'} | 📸 Capturas guardadas")
            except Exception as e:
                print(f"✅ Muestra | Error: {self.error_distance:.1f}px | Angular: {angular_error:.2f}° | NSS: {nss:.3f} {'🟢' if nss > 1.0 else '🔴'} | ❌ Error al guardar foto: {e}")
            
            self.waiting_for_click = False
            # Keep the point visible until the next sample.

def parse_args():
    parser = argparse.ArgumentParser(description="ADAS y estimación de mirada en tiempo real")
    parser.add_argument("--lr-device-id", help="MXID de la cámara de carretera")
    parser.add_argument("--pro-device-id", help="MXID de la cámara del conductor")
    parser.add_argument(
        "--allow-auto-device-order", action="store_true",
        help="Permitir la asignación insegura LR/Pro usando el orden USB",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directorio de salidas opcionales")
    parser.add_argument(
        "--extrinsics",
        default="calibracion_extrinseca/extrinsics_pro_to_lr_no_mirror.json",
        help="JSON con R y T para transformar de OAK-D Pro a OAK-D LR",
    )
    parser.add_argument("--yolo-model", default="models/yolov8n-seg.pt",
                        help="Peso de segmentación compatible con Ultralytics")
    parser.add_argument("--gaze-model", default="models/resnet101_v2.pt",
                        help="Checkpoint ResNet-101 de estimación de mirada")
    parser.add_argument("--face-detector-model", default="models/face-detection-retail-0004.blob",
                        help="Blob OpenVINO del detector facial para DepthAI")
    parser.add_argument("--landmark-model", default="models/facial-landmarks-35-adas-0002.blob",
                        help="Blob OpenVINO de 35 landmarks faciales para DepthAI")
    parser.add_argument("--record-video", action="store_true", help="Guardar video procesado")
    parser.add_argument("--save-reports", action="store_true", help="Guardar reportes JSON")
    parser.add_argument("--save-captures", action="store_true", help="Guardar imágenes que pueden contener rostros")
    parser.add_argument("--save-dataset-frames", action="store_true", help="Permitir guardar frames para YOLO")
    return parser.parse_args()


def main():
    """Main function"""
    try:
        adas = GazeAwareADAS(parse_args())
        adas.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
