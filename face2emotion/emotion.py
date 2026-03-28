import logging
import cv2
import numpy as np
from .schema import EmotionResult

log=logging.getLogger("face2emotion")
#情绪种类
all_labels=["Anger", "Contempt", "Disgust", "Fear","Happiness", "Neutral", "Sadness", "Surprise"]
lowers={"Anger": "angry", "Contempt": "contempt", "Disgust": "disgust","Fear": "fear",   "Happiness": "happy",   "Neutral": "neutral","Sadness": "sad", "Surprise": "surprise"}

ci=all_labels.index("Contempt")
ni=all_labels.index("Neutral")
#ema新帧权重
w_new=0.6
w_old=0.4
min_size=30
#左右眼角眼尾位置
l_eye=[33,133]
r_eye=[362,263]

def _eye_centers(lm,w,h):
    #左右眼瞳孔中心坐标
    #lm:人脸关键点列表 w:图像宽度 h:图像高度
    try:
        lx=sum(lm[i].x*w for i in l_eye)/2
        ly=sum(lm[i].y*h for i in l_eye)/2
        rx=sum(lm[i].x*w for i in r_eye)/2
        ry=sum(lm[i].y*h for i in r_eye)/2
        return (lx,ly),(rx,ry)
    except Exception:
            return None
    #仿射变换增加模型泛化能力        
def _align(bgr,lm):
        h,w=bgr.shape[:2]
        pts=_eye_centers(lm,w,h)
        if not pts:
            return None
        (lx,ly),(rx,ry)=pts
        #双眼连线与水平线的夹角
        angle=np.degrees(np.arctan2(ry-ly, rx-lx))
        cx,cy=(lx+rx)/2,(ly+ry)/2
        M=cv2.getRotationMatrix2D((cx,cy),angle,1.0)
        return cv2.warpAffine(bgr,M,(w, h),flags=cv2.INTER_LINEAR)    
    #仿射变换执行过程   
def _preprocess(bgr,lm=None):
        h,w=bgr.shape[:2]
        if h<min_size or w<min_size:
            return bgr
        if lm is not None:
            aligned=_align(bgr,lm)
            if aligned is not None:
                return aligned
        return bgr        
#将轻蔑(Contempt)情绪去除
#Contempt样本少,容易混淆,所以合并来提升稳定性
def _to7(p8):
        p=p8.copy()
        p[ni]+=p[ci]
        out={};tot=0.0
        for i,labels in enumerate(all_labels):
            if labels=="Contempt":continue
            v=float(p[i])
            out[lowers[labels]]=v
            tot+=v
    #因为少了一个，所以重新进行归一化    
        if tot>1e-9:
            out={k:v/tot for k, v in out.items()}
        return out
class EmotionRecognizer:
    def __init__(self):
        self.model=None
        self.mesh=None
        self.buf:dict[int,dict[str,float]]={}
    def _get_model(self):
        if self.model is None:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            self.model=HSEmotionRecognizer(model_name="enet_b2_8")
        return self.model
    #参数作用：    
    #static_image_mode:视频流模式（连续帧）
    #max_num_faces:最多检测1张脸
    #refine_landmarks:不细化关键点（468点）
    #min_detection_confidence:检测置信度阈值
    #min_tracking_confidence:跟踪置信度阈值
    def _get_mesh(self):
        if self.mesh is None:
            import mediapipe as mp
            self.mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,refine_landmarks=False,min_detection_confidence=0.4,min_tracking_confidence=0.4)
        return self.mesh
    #从 BGR 图像中提取人脸的 468 个关键点坐标，包含异常处理和日志记录    
    def _landmarks(self,bgr):
       try:
           #颜色空间转换
           res=self._get_mesh().process(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
           if res.multi_face_landmarks:#检测是否有人脸
               return res.multi_face_landmarks[0].landmark
       except Exception as e:
           log.debug("landmark err: %s",e)
       return None   
        
    def _smooth(self,tid,dist):
        if tid<0: return dict(dist)
        old=self.buf.get(tid)
        if old is None:
            self.buf[tid]=dict(dist)
            return dict(dist)
        # 指数加权平均：平衡响应速度与输出稳定性
        blended = {k: w_new*dist.get(k,0.) + w_old*old.get(k,0.) for k in dist}
        self.buf[tid]=blended
        return blended
        
    def predict(self,bgr,track_id=-1):
        if bgr is None or bgr.size==0:
            return self._fallback(track_id)
        try:
            lm =self._landmarks(bgr)
            img= _preprocess(bgr, lm)
            _, scores=self._get_model().predict_emotions(img,logits=False)
            arr =np.asarray(scores, dtype=np.float32).ravel()
            sm= self._smooth(track_id,_to7(arr))
            k, v=max(sm.items(), key=lambda x: x[1])
            return EmotionResult(label=k,score=float(v))
        except Exception as e:
            log.warning("predict failed tid=%s: %s",track_id,e,exc_info=True)
            return self._fallback(track_id)

    def forget(self,active:set):
        for k in [k for k in self.buf if k not in active]:
            del self.buf[k]

    def _fallback(self, tid):
        h =self.buf.get(tid)
        if h:
            k, v= max(h.items(), key=lambda x: x[1])
            return EmotionResult(label=k, score=float(v))
        return EmotionResult(label="unknown",score=0.0)
    
