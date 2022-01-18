import Video
import Annotation

_ = Video.VideoCompressor('./videos', './npy_videos')
_ = Annotation.AnnotationsCompressor('./alignment', './npy_alignment')