# Project 5

### subject : 항공권 예측 [ML & Prototype]

<img src='https://github.com/Choe-minsung/project-study/blob/ca5d45d60d0142fd2e22bdcbc7b869d503cbe826/P5/src/main.png' width='1100'/>

- duration : 2023.11.14 ~ 2023.11.27
- stack : Python (Jupyter Notebook)
- member : **PM** 1 / 기술 3
- my role : **PM**

#### pipeline
1. 데이터탐색, 전처리
2. ML modeling  
    1. **LinearRegression**  
    2. **KNeighborsRegressor**  
    3. **DecisionTreeRegressor** (’max_depth’ : 10)  
    4. **RandomForestRegressor** (’max_depth’ : 10, ‘n_estimators : 100)  
    5. **GradientBoostingRegressor** (’max_depth’ : 10, ‘n_estimators’ : 100, ‘learning_rate’ : 0.1**)**  
    6. **XGBRegressor** (’max_depth’ : 10)  
    7. **LGBMRegressor** (’verbose’ : -100)  
    8. **CatBoostRegressor** (**’**learning_rate’ : 0.1, ‘depth’ : 10)  
3. model selection & model tuning  
4. **Prototype** 구현  
    
    4-1. preprocessing 함수 생성
    
    4-2. 가격예측 함수 생성
    
    4-3. 통계분석 df 생성
    
    4-4. layout 구성


<img src='https://github.com/Choe-minsung/project-study/blob/3bfaedf59db71a6db35e82558886efe8af1f5273/P5/src/feature_map.png' width='700'/>


<img src='https://github.com/Choe-minsung/project-study/blob/12f5aec14b1ae39b00ea4f1270eab140566ccb07/P5/src/pred_airfare.png' width='700'/>

