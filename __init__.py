from .nodes.save_load_pose import TSSavePoseDataAsPickle, TSLoadPoseDataPickle
from .nodes.openpose_smoother import KPSSmoothPoseDataAndRender, KPSSmoothPoseKeypointAndRender
from .nodes.rename_files import RenameFilesInDir
from .nodes.color_match import TSColorMatchSequentialBias


NODE_CLASS_MAPPINGS = {
    "TSSavePoseDataAsPickle": TSSavePoseDataAsPickle,
    "TSLoadPoseDataPickle": TSLoadPoseDataPickle,
    "TSPoseDataSmoother": KPSSmoothPoseDataAndRender,
    "TSPoseKeypointSmoother": KPSSmoothPoseKeypointAndRender,
    "TSRenameFilesInDir": RenameFilesInDir,
    "TSColorMatch": TSColorMatchSequentialBias,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSSavePoseDataAsPickle": "TS Save Pose Data",
    "TSLoadPoseDataPickle": "TS Load Pose Data",
    "TSPoseDataSmoother": "TS Pose Data Smoother",
    "TSPoseKeypointSmoother": "TS Pose Keypoint Smoother (DWPose/OpenPose)",
    "TSRenameFilesInDir": "TS Rename Files In Dir",
    "TSColorMatch": "TS Color Match",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
