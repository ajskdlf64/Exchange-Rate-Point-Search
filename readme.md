# USD 환율 가격 예측 모형

### 1. Data Loading

- **1.1 Sliding Window Dataset**

  - 시계열 데이터의 경우 장기간의 데이터를 사용하는 것보다 **영향력이 더 강한 최근 데이터**만 사용하여 모델을 만드는 것이 더 성능이 좋음.'
  - 모델의 **일반화(Generalization)** 확보를 위해서 각 월별로 1개의 모델, 총 12개의 모델을 생성

    ![figure](./figure/figure01.png)

- **1.2 Train & Valid & Test**
  
  - 각 Dataset 별로 앞에서부터 **11개월간을 Train**으로 사용하고, **12번째 월은 Valid**로 사용, 마지막 **13번째 열은 Test**로 사용한다.
  - Train : Valid : Test = 84 : 8 : 8

<br><br>

### 2. Feature Engineering

- **2.1 asfasfasfasfasdf**
  - 가나다라마바사
  - 가나다라마바사
  - 가나다라마바사

- **2.2 asfasfasfasfasdf**
  - 가나다라마바사
  - 가나다라마바사
  - 가나다라마바사

<br><br>

### 3. Model 1 : Deep Neural Network
- **3.1 모델 설명**
   - 3.1.1 Keras를 활용하여 2층 구조의 DNN 모델 설계.
   - 3.1.2 Overfitting 방지를 위해서, 2개의 층만 설정, L1, L2 규제 사용, Dropout
   - 3.1.3 활성화함수 : ReLU, Loss : RMSE, Epoch : 1000, batch_size : 200
  
- **3.2 결과 요약**
  - 3.2.1 Set1만 돌렸을 때의 코드 **-> [Set1 Code](https://github.com/ajskdlf64/Exchange-Rate-Point-Search/blob/master/Code/02.%20DNN%20Set1.ipynb)**
  - 3.2.2 Set1 ~ Set12를 모두 돌렸을 때의 코드 **-> [Set1 ~ Set12 Code](https://github.com/ajskdlf64/Exchange-Rate-Point-Search/blob/master/Code/03.%20DNN%20Set1%20_%20Set12.ipynb)**
  
  <br><br>

### 4. Model 2 : Recurrent Neural Network (LSTM, GRU)
- **4.1 모델 설명**
   - 4.1.1 가나다라마바사
   - 4.1.2 가나다라마바사
   - 4.1.3 가나다라마바사
  
- **4.2 결과 요약**
  - 4.2.1 가나다라마바사
  - 4.2.2 가나다라마바사
  
  
  ### 참고사항
   - 모델 안정성 확보를 위해 시뮬레이션 100번 진행, 1번당 약 3분 -> 100번 약 300분, 5시간
   - 분류모델도 고려...?
   - LSTM, GRU도 함께 고려...
   - 하이퍼 파라미터 그리드 서치
   - 매달 최저가로 구매하면 얼마나 절약 가능할까...?
   - 지금은 과거 5일치만 -> 좀 더 늘려볼까...?
  
   <br><br>

### 5. Model 3 : Pattern Recognize with LSTM, GAN
- **5.1 모델 설명**
   - 5.1.1 가나다라마바사
   - 5.1.2 가나다라마바사
   - 5.1.3 가나다라마바사
  
- **5.2 결과 요약**
  - 5.2.1 가나다라마바사
  - 5.2.2 가나다라마바사

   <br><br>

### 참고사항
 - 모델 안정성 확보를 위해 100번 시뮬레이션(1번당 약3분, 100번이면 300분,5시간)
 - 분류모델로 변경...?
 - LSTM, GRU도 고려
 - Hyper Parameter 그리드 서치
 - 매달 최저가로 구매시 얼마나 절약 가능할까?
 - 지금은 과거 5일치만, -> 좀 더 늘려볼까?
