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

#### Batch Normalization - Layer Normalization 

Normalization을 하는 이유 - 모든 데이터들의 스케일을 동일하게 만들어서 각 feature 값들이 동등한 중요도를 가지도록 하는 작업이다. 특정하게 치우진 데이터들을 넓게 분포하는 형태로 만들 수 있기 때문에 학습적인 측면에서 더 효율적이라고 말할 수 있다. 

![[Pasted image 20240813142412.png]]
![[Pasted image 20240813143004.png]]

![[Pasted image 20240813143014.png]]



![[Pasted image 20240813143023.png]]



![[Pasted image 20240813143032.png]]

- Feature 별로 mean, std 추출하는 Batch Normalization - batch size에 의존적임 
- 각 데이터 별 mean, std 추출하는 Layer Normalization - batch size에 의존적이지 않음

##### Input and Output parameterization





##### 다른 결과들과 비교하는 것에 대해서

