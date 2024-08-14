> 핵심이 되는 GPT, SelfAttention, Block 클래스를 먼저 살펴보면 

##### 인자

- n_embd : 임베딩하려는 차원을 의미한다. Q K V의 차원이라고 생각하면 되겠다
- n_head : head의 개수를 의미한다. 관점의 개수라고 생각하자 
- attn_pdrop, resid_pdrop : attention, residual connection을 진행할 때 dropout을 진행하는데 이 때의 확률값을 갖는 인자라 고 생각하자

- proj : projection 을 마지막으로 진행하면서 결과적으로 입, 출력의 shape(정확히 말하면 차원)을 맞춰주는 역할을 수행한다

#### forward() 함수

self attention을 실질적으로 수행하는 함수라고 생각하자 
input : B(Batch_size) x T(Sequence_Length) x C(Embedding dimension)


헤드 차원을 배치 차원과 가까이 둬야 한다 - 병렬 연산, 계산의 효율화!!!
근접한 위치에 둬야 배치 안의 여러 헤드들에 대해서 병렬적인 계산을 적용할 수 있기 때문이다. 나란히 배열을 해야 한다~ 라는 뜻으로 이해하면 되겠다.

#### ==self.attn_drop, resid_drop에 대해서==

##### attention dropout

드롭아웃이 적용되면 Attention Score의 일부 요소가 무작위로 0이 된다. 임베딩 차원에서 영향력을 끼치지는 않고 단순히 output에 대해서 일부 요소의 값을 0으로 만든다. 일부를 비활성화 하고 나머지 값들은 합이 1이 되도록 조정이 이뤄진다. 

(효과) 특정 토큰 간의 관계에 의존하는 것을 방지할 수 있고 더 다양한 패턴을 파악할 수 있다. 
-> 일부러 변화를 가함으로써 더 다양한 패턴을 학습하기 위한 과정이라고 생각하자. 오버피팅을 방지할 수 있는 효과도 가지고 있다

드롭된 가중치는 0이되고 나머지는 다시 정규화된다.

##### residual connection dropout

code의 내용을 빌려서 적용해보면 self.proj(y) projection을 진행하고 난 output이 있다고 가정하자.
output의 값 중에서 일부 요소를 비활성화시킨다고 생각하면 되겠다. 이 역시 다양한 패턴의 학습에 도움을 준다. 특정 경로에만 의존하지 않도록, 다양한 경로를 통해 학습하도록 한다 (일반화 하기 위한 과정이다)

결과적으로 nn.Dropout()을 적용함으로써 네트워크의 일반화 성능을 향상, 정보를 무작위로 제거함으로써 학습을 더 robust하게 만들어준다고 생각하면 되겠다.


##### ==y.transpose(1, 2).contiguous().view(B, T, C) 를 하는 이유==

단순히 차원을 변환하려 한다면 reshape(B, T, C) 로 할 수 있지 않나? 하는 궁금증이였다

reshape()을 사용해도 되지만 개발자의 의도를 더 명확히 전달하기 위해서 contiguous()를 활용해서 연속적인 메모리 레이아웃을 갖는 텐서를 반환하고 view()를 활용해서 차원을 변환한다. 이 때 view() 함수는 연속적인 메모리 레이아웃을 갖는 텐서에 대해서만 작동한다

reshape()의 경우 내부에서 contiguous()가 발생할 수 있는데, 이로 인해서 추가적인 메모리를 사용할 수 있다는 점도 알아둬야 한다. 

따라서 메모리 레이아웃에 대한 제어를 더 명확하게 전달할 수 있는 장점을 가지고 있다 


##### Transfuser 관련 nn.Dropout() 관련해서 

Embedding - Self Attention - Residual connection 모든 경우에 대해서 드롭 아웃을 적용했다. 일반화 성능을 높이려 이 3가지 구간에 적용했구나~ 하는 정도로 이해하고 넘어가자

##### Layer Normalization을 하는 이유에 대해서 

- 안정적인 학습 - 특히 트랜스포머 구조 내에서는 잔차 연결에 있어서 중요한 역할을 수행한다 


##### Model내 하위 모듈에 대해서 (레이어에 대해서) 가중치 및 bias 고정하고 싶다면

- self.apply()함수를 적용하면 된다 
- if isInstance(module, nn.Linear or nn.LayerNorm):
	- module.weight.data.normal_(mean, std) : 전달된 mean, std를 따르는 정규분포를 갖는 가중치로 만들어주세요 라는 함수가 되겠다
	- module.bias.data.zero_()
- module.bias.data.zero_() : bias를 0으로 (편향은 반영하지 않겠다는 말)
- module.weight.data.fill_() - 가중치를 인자로 전달된 값으로 통일 


기초적인 초기화 절차 중 하나라 매번 모델을 돌릴 때마다 가중치가 랜덤으로 적용되서는 안 된다 (애초에 통일된 조건에서 같이 반복했을 때 변화를 살피는게 올바른 과정). 따라서 self.apply() 를 적용해서 가중치와 편향의 초기화를 담당하는 함수를 호출하는 방식으로 진행하면 되겠다. 


##### bz : batch_size를 의미한다 

torch.cat(A, B, dim = 1)


##### vanilla multi-head masked self-attention

가장 기본적인 형태의 self-attention을 적용했다고 생각하면 되겠다. 트랜스포머 모델의 기본 형태의 self-attention을 의미한다. 입력 요소 그리고 다른 요소들 간의 상호작용을 계산한다. 여러 헤드로 나눠 병렬로 처리한다 - masked는 시퀀스 위치의 미래 요소를 볼 수 없도록 해서 순차적으로 예측이 가능하도록 만드는 것을 의미한다. 

---

## ImageCNN (Image encoder network)

out_features -> 최종적으로 출력할 벡터의 피처 개수를 의미한다 
self.normalize = normalize (데이터 정규화 여부를 결정한다 - 평균과 표준편차로 조정한다)
- 인자로 전달되는 normalize가 TRUE인 경우에는 정규화 방식을 표준 데이터셋에 맞게 정규화할 수 있다. 
self.features = timm.create_model(architecture, pretrained=True)
- architecture로 전달한 모델을 사용하겠다는 말 

\_\_init_\_() : CNN 모델을 가지고 오고 + 최종적인 fully connected 레이어 제거 (최종적인 작업을 하는 과정을 수정하기 위함. 말 그대로 중간 과정만 사용하겠다는 뜻으로 이해하면 되겠다)

그대로 가지고 오는 경우에는 stem을 활용해서 기존의 레이어 정보를 그대로 가지고 오면 되겠다. 통일된 코드 구조를 적용할 수 있다.


## LidarEncoder(nn.Module)

다른 모델 정보를 그대로 적용한다고 생각하기 


---

### Base questions


#### Batch Normalization - Layer Normalization 

Normalization을 하는 이유 - 모든 데이터들의 스케일을 동일하게 만들어서 각 feature 값들이 동등한 중요도를 가지도록 하는 작업이다. 특정하게 치우진 데이터들을 넓게 분포하는 형태로 만들 수 있기 때문에 학습적인 측면에서 더 효율적이라고 말할 수 있다. 

![[Pasted image 20240813142412.png]]
![[Pasted image 20240813143004.png]]

![[Pasted image 20240813143014.png]]



![[Pasted image 20240813143023.png]]



![[Pasted image 20240813143032.png]]

- Feature 별로 mean, std 추출하는 Batch Normalization - batch size에 의존적임 
- 각 데이터 별 mean, std 추출하는 Layer Normalization - batch size에 의존적이지 않음





--- 
##### Input and Output parameterization


#### Waypoint prediction by GRU  - code 확인



##### Auxilary loss functions 

- 2D depth estimation (픽셀의 깊이 정보를 예측하는 작업 - 카메라로부터 거리를 추정한다)
- 2D semantic segmentation - cross-entropy-loss (시멘틱 클래스를 구분한다)

###### HD Map - 고해상도 사진을 말한다. 

BEV segmentation mask (3가지의 경우로 분리할 수 있겠다) - 도로(주행 가능한 영역) + 차선 표시(차선) + 기타로는 주행이 불가능한 영역을 말하겠다.
LiDAR data - feature map to 2D image data

###### Bounding box : 차량의 위치를 탐색하기 위함 (CenterNet decoder)

키포인트 추정 - 장면 내 다른 차량의 위치를 탐색
BEV  특징을 사용해서 Convolution decoder 적용해서 위치 맵을 예측합니다. 64x64 예측하여 차량을 탐지한다

Coarse orientation - 예측하기 위해서 실제 차량의 상대적인 요 값을 12개의 30도 크기의 구간으로 이산화해서 클래스를 각 픽셀에서 12채널 분류 레이블을 활용하여 예측한다 

==마지막으로 회귀 맵을 예측합니다 (차량 크기, 위치 오프셋, 방향 오프셋)==

- 위치 오프셋 : 입력보다 낮은 해상도로 위치 맵을 예측할 때 발생하는 양자화 오류를 보정하는데 사용됩니다.
- 위치 맵, 방향 맵, 그리고 회귀 맵은 포컬 손실, 교차 엔트로피 손실 및 L1 손실을 사용하여 학습된다.

주변의 다른 차량을 정확하게 인식하고 추적할 수 있도록 하는 복잡한 예측 과정이다




## 3.5 Controller 

We use two PID controllers for lateral and longitudinal control to obtain steer, throttle, and brake values from the predicted waypoints, {w t } Tt=1
-> 측면(lateral) 제어 그리고 종방향 제어를 위한 PID 제어기 2개를 사용합니다. 

- Creeping : 예상되는 적신호 대기 시간보다 긴 55초동안 움직이지 않는 경우 PID 제어기의 목표 속도를 4m/s로 설정해서 차량이 잠시 (1.5초) 동안 앞으로 움직이도록 한다. 
	- 이는 모방 학습에서 관찰된 관성 문제를 해결하기 위함이다. 차량이 정지한 경우, 훈련 데이터에서도 정지할 확률이 매우 높기 때문입니다. 에이전트는 정지한 후 다시 주행을 시작하지 않는 문제가 발생할 수 있다.

- Safety Heruistic : 차량이 밀집된 경우 크리핑을 적용하게 되면 Collision이 발생할 수 있다. 따라서 차량 앞의 작은 직사각형 영역에서 LiDAR 신호가 감지되면 크리핑 동작을 무시하는 안전 검사를 구현합니다. 크리핑 중에서는 필수적이지만 정규 주행 중에도 적용해서 안전성을 높일 수 있다. 
	- 크리핑 상태 그리고 정규 주행 중에서도 안전 휴리스틱을 적용한 영향을 4.11절에서

#### Auxilary Loss Functions

waypoint loss를 제외하고도 4개의 보조 Loss Function을 구현했다. depth-prediction and semantic segmentation from the image branch; HD map detection and vehicle detection from the BEV branch.


##### 3.7 Latent TransFuser

> 감이 잘 오지를 않음. 아래는 paper의 본문 내용을 그대로 가지고 와봤다. 각각의 부분에 대해서는 code를 확인해보기로 

- CILRS : 이미지 기반 베이스라인을 말한다. 시각적 특징을 활용해서 차량 제어를 예측한다. 
- 이미지 전용 버전의 Latent TransFuser를 소개합니다. 
- 2채널 LiDAR BEV 히스토그램 입력을 동일한 크기의 2채널 위치 인코딩으로 대체합니다. BEV 좌표 프레임에서 이미지 특징을 융합하여 BEV로의 주의 기반 투영을 수행한다
- LiDAR 데이터 대신 이미지 기반의 위치 인코딩을 사용해서 자율주행 시스템에서 더 효율적인 학습이 가능하도록 만들어준다.

입력 데이터의 변경 : CILRS와 달리 2채널 LiDAR BEV 히스토그램 입력을 
==동일한 크기의 2채널 위치 인코딩으로 대체.==

이미지 기반 입력을 사용함에도 LiDAR의 안전성을 강화하기 위한 입력을 포함한다. 
HD 맵, 바운딩 박스를 예측하는 LiDAR 브랜치를 사용해서 BEV 좌표 프레임 내에서 주의 기반 투영을 수행한다. 



#### HD 맵, 바운딩 박스를 예측하는 LiDAR 브랜치를 사용

HD맵은 자율주행 차량에서 사용하는 고정밀 지도를 의미한다. 매우 상세하고 정확한 정보를 가지고 있다. 자신이 도로 어느 위치에 있는지를 파악하여 차선 변경 및 회전 등 복잡한 주행 동작을 보다 정밀하게 수행할 수 있다

바운딩 박스는 객체 탐지를 위해 사용되는 기본적인 컴퓨터 비전 기법이다. 바운딩 박스는 이미지나 비디오 프레임 내에서 객체의 위치 그리고 크기를 직사각형으로 표시하는 것을 말한다. 

위의 2가지 요소를 연계해서 자율 주행 차량이 도로 환경을 이해하고 안전하게 주행하기 위해서 보완적인 역할을 수행한다고 보면 되겠다. HD맵은 차량이 도로의 구조 그리고 중요한 표지물을 정확히 파악할 수 있게 해주는 반면, 바운딩 박스는 실시간으로 주변의 이동하는 객체들을 탐지하고 추적할 수 있다. 



---

## Experiments

experimental setup, compare the driving performance of TransFuser against several baselines, visualize the attention maps of TransFuser and present ablation studies to highlight the importance of different components of our approach

##### Task 

미리 정의된 경로를 따라 네비게이션 한다. 미리 정의된 위치에서 초기화되고 에이전트가 다양한 종류의 적대적인 상황을 처리하는 능력을 테스트 한다.

```markdown
훈련 데이터는 2FPS(초당 프레임 수)로 저장되며, 총 228,000개의 프레임으로 구성된 대규모 데이터셋이 생성됩니다. 이 데이터셋은 자율주행 시스템을 훈련시키는 데 사용됩니다.
```

##### 4.3 Expert

