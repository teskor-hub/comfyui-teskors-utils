from .nodes.save_load_pose import TSSavePoseDataAsPickle, TSLoadPoseDataPickle
from .nodes.openpose_smoother import KPSSmoothPoseDataAndRender
from .nodes.load_video_batch import LoadVideoBatchListFromDir
from .nodes.rename_files import RenameFilesInDir
from .nodes.color_match import TSColorMatchSequentialBias


NODE_CLASS_MAPPINGS = {
    "TSSavePoseDataAsPickle": TSSavePoseDataAsPickle,
    "TSLoadPoseDataPickle": TSLoadPoseDataPickle,
    "TSPoseDataSmoother": KPSSmoothPoseDataAndRender,
    "TSLoadVideoBatchListFromDir": LoadVideoBatchListFromDir,
    "TSRenameFilesInDir": RenameFilesInDir,
    "TSColorMatch": TSColorMatchSequentialBias,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSSavePoseDataAsPickle": "TS Save Pose Data (PKL)",
    "TSLoadPoseDataPickle": "TS Load Pose Data (PKL)",
    "TSPoseDataSmoother": "TS Pose Data Smoother",
    "TSLoadVideoBatchListFromDir": "TS Load Video Batch List From Dir",
    "TSRenameFilesInDir": "TS Rename Files In Dir",
    "TSColorMatch": "TS Color Match",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
