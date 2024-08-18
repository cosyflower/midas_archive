
object detection 은 무엇인가?? 에 대해서 알아봅시다

- classification
- localization

이 2가지를 모두 수행하는 것이 object detection 이라고 생각하면 되겠다 

#### 1 stage 그리고 2 stage 

- 물체를 먼저 탐지하고
- 탐지된 물체가 어떤 물체인지를 classification 하는 과정
이 2개의 과정을 한번에 처리하는 모델이 있는 반면, 서로 다른 stage에서 처리하는 모델도 존재한다. 이러한 차이를 구분하기 위한 개념으로 인지하면 되겠다 


![[Pasted image 20240814212617.png]]


![[Pasted image 20240814212600.png]]


#####  Bounding box - 이미지 내에서 물체 전체를 작게 그린 가장 작은 직사각형

- width
- height 

##### confidence score -  이미지 내에서 찾은 bounding box안에 물체가 있을 확률

학습이 잘 된 모델일수록 confidence score이 높아야 한다

##### ==sliding window - 이미지 내 적당한 크기의 영역을 정해서==

영역을 이동(sliding)하면서 알고리즘을 적용하는 방식을 의미한다 


##### anchor box : 특정 사이즈 혹은 비율로 미리 정의된 box


![[Pasted image 20240814213129.png]]

미리 기정의된 박스라고 생각하면 되겠다 


#### NMS (Non-Maximum Suppression)

검출된 bounding box 중에서 비슷한 위치 혹은 Object일 확률이 낮은 box들을 제거하고 가장 적합한 box를 찾는 과정을 NMS라고 한다 

![[Pasted image 20240814213243.png]]


### Object detection 관련 Metric

![[Pasted image 20240814213439.png]]

실제 클래스 그리고 판별한 결과를 서로 비교한다고 생각하면 되겠다 
같은 경우, 서로 다른 경우에 대해서 어떤 명칭이 존재하는지를 구분하면 되겠다 
-> 다르게 예측한 경우 명칭이 어떤게 있냐면 - false negative, false positive이 존재한다. 이 때 뒤에 붙여지는 용어는 예측을 어떻게 했냐를 설명하는 부분이라고 생각하면 되겠다
- false negative : 실제는 positive 이지만 예측한 결과 negative로 예측한 경우를 의미하겠다   


#### confusion matrix를 기반으로 Accuracy, Recall, Precision

![[Pasted image 20240814213629.png]]


#### 기준의 차이 : detected box vs object 

- detected box : precision
- recall : object 

![[Pasted image 20240814214247.png]]

각각이 중요한 시기가 서로 다르다.

#### IOU (Interaction over Union)

![[Pasted image 20240814223147.png]] 

#### MAP and IOU

![[Pasted image 20240814224219.png]]

- Trade off 관계를 유지하는 정도로 알고있자



## NMS 알고리즘 

Non max suppression 알고리즘에 대해서 알아보자 

- 검출된 Bounding box. 여러 개가 검출된 상황이라고 가정하자. 비슷한 위치에 있거나, Object일 확률이 낮은 box들을 제거하고, 가장 적합한 box를 찾는 것을 Non-Maximum suppression 이라고 했었다
- confidence score이 낮은 박스들을 제외하고 가장 최적의 박스만을 남긴다고 생각하면 되겠다


![[Pasted image 20240814230047.png]]


-> 핵심은 3번째 단계가 되겠다 - 순차적으로 Confidence score을 정렬한 다음. 가장 높은 점수의 박스를 기준으로 삼는다. 여기서 기준으로 삼는다는 것은 해당 박스의 class와 동일하면서 동시에 iou_threshold 이상인 박스, 이러한 조건들을 모두 만족해야 한다는 것을 의미한다

-> 참고로 위의 예시에서는 CAR 클래스를 만족하는 다른 bounding box로 0.8인 친구가 있기 때문에 0.9 그리고 0.8인 (confidence score) bounding box가 nms 알고리즘의 결과로 보면 되겠다

#### Intersection over union 의 값이 크면 클수록 완벽하게 detection 했다는 것

![[Pasted image 20240814230424.png]]


![[Pasted image 20240814230745.png]]

  

#### Object Detection Dataset

- coco (Common objects in context)
- pascal2
- kitti-360
- udacity (self-driving dataset)
- open images dataset extensions

https://deepbaksuvision.github.io/Modu_ObjectDetection/posts/02_00_Datasets_for_Object_Detection.html

위의 링크에 들어가서 확인하기!!

