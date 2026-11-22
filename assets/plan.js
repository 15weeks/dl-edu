/* DL-Edu 15장 계획. 차례 page, 왼쪽 메뉴, PLAN_KR.md / PLAN_EN.md가 모두 이 파일 하나에서 나온다
   DL-Edu 15-chapter plan. The home pages, the left menu and PLAN_KR.md / PLAN_EN.md all come from this one file */
"use strict";

/* 단계: 차례 page 지도의 열. "|"는 줄바꿈
   Stages: the columns of the home-page map. "|" is a line break */
const DLEDU_STAGES = [
  { ko: "① 출발", en: "① Starting out", ch: [1, 2] },
  { ko: "② 신경망", en: "② Neural networks", ch: [3, 4] },
  { ko: "③ 잘 학습시키기", en: "③ Training well", ch: [5, 6, 7] },
  { ko: "④ 실전", en: "④ In practice", ch: [8, 9] },
  { ko: "⑤ 그림을 보는 구조", en: "⑤ Models for|images", ch: [10, 11, 12] },
  { ko: "⑥ sequence와 생성", en: "⑥ Sequences and|generation", ch: [13, 14, 15] }
];

/* 장마다: file(언어 표시 없는 이름), name(메뉴·지도에 쓰는 짧은 장 이름), topic(다루는 것), 목표, 절, 그림, 대응 Udemy 강의
   Per chapter: file (base name without language), name (short chapter name for the menu and map), topic (what it covers), goal, sections, figures, matching Udemy lectures
   절과 그림은 [한글, 영어] 짝 / sections and figures are [Korean, English] pairs */
const DLEDU_PLAN = [
  {
    n: 1, file: "ch01-intro", ready: true,
    name: { ko: "기초", en: "Foundations" },
    topic: { ko: "수학·통계·코드", en: "Math, statistics and code" },
    goal: { ko: "딥러닝이 쓰는 도구(vector, matrix, softmax, cross-entropy, 평균·분산, t-test, NumPy·PyTorch)를 손으로 계산하고 그림으로 움직여 본다.", en: "Compute by hand, and move in pictures, the tools deep learning uses: vectors, matrices, softmax, cross-entropy, mean and variance, the t-test, NumPy and PyTorch." },
    sections: [
      ["세상을 숫자로: scalar, vector, matrix, tensor", "Turning the world into numbers: scalar, vector, matrix, tensor"],
      ["transpose와 shape", "Transpose and shape"],
      ["dot product", "Dot product"],
      ["matrix 곱", "Matrix multiplication"],
      ["softmax", "Softmax"],
      ["log와 exp", "Log and exp"],
      ["entropy와 cross-entropy", "Entropy and cross-entropy"],
      ["min/max와 argmin/argmax", "Min/max and argmin/argmax"],
      ["평균, 분산, 표준편차", "Mean, variance, standard deviation"],
      ["random sampling과 seed", "Random sampling and seeds"],
      ["t-test", "The t-test"],
      ["Python, NumPy, PyTorch 첫걸음 (broadcasting 포함)", "First steps in Python, NumPy, PyTorch (with broadcasting)"],
      ["미분 맛보기", "A first taste of derivatives"],
      ["정리와 약자 사전", "Summary and dictionary"]
    ],
    figures: [
      ["8×8 pixel 사진 칠하기 ↔ 숫자 matrix", "Paint an 8×8 picture ↔ its number matrix"],
      ["transpose 뒤집기 animation, matrix 곱 칸 고르기", "Transpose flip animation; pick a cell of a matrix product"],
      ["두 vector 끌기: dot product, 각도, 그림자", "Drag two vectors: dot product, angle, projection"],
      ["softmax temperature, entropy·cross-entropy 곡선", "Softmax temperature; entropy and cross-entropy curves"],
      ["분산 막대, 표본 평균 histogram, t-test 시뮬레이터, 할선 → 접선", "Variance bars, sample-mean histogram, t-test simulator, secant → tangent"]
    ],
    udemy: ""
  },
  {
    n: 2, file: "ch02-gradient-descent", ready: true,
    name: { ko: "Gradient descent", en: "Gradient descent" },
    topic: { ko: "미분, model이 배우는 방법", en: "Derivatives and how models learn" },
    goal: { ko: "학습 = loss를 줄이는 쪽으로 parameter를 조금씩 고치기. 이것을 미분과 gradient descent로 이해한다.", en: "Learning = nudging parameters in the direction that lowers the loss. Understand this through derivatives and gradient descent." },
    sections: [
      ["model이 '배운다'는 것: parameter, loss, 학습 고리", "What 'learning' means: parameters, loss, the training loop"],
      ["미분의 직관과 다항식 미분", "Derivative intuition and polynomials"],
      ["미분으로 최솟값 찾기", "Finding minima with derivatives"],
      ["product rule과 chain rule", "Product rule and chain rule"],
      ["1D gradient descent", "Gradient descent in 1D"],
      ["learning rate와 시작점 (고정 vs 바뀌는 learning rate)", "Learning rate and starting value (fixed vs dynamic)"],
      ["local minima와 saddle point", "Local minima and saddle points"],
      ["2D gradient descent와 gradient vector", "Gradient descent in 2D and the gradient vector"],
      ["vanishing·exploding gradient 맛보기", "Vanishing and exploding gradients (preview)"]
    ],
    figures: [
      ["data에 직선 맞추기 + loss 곡선 L(w): 한 걸음씩 학습", "Fitting a line to data + the loss curve L(w): learning one step at a time"],
      ["기울기 그래프 그리기: 접선이 움직이며 f′를 그린다", "Drawing the slope graph: the moving tangent traces f′"],
      ["f′의 부호로 칠한 곡선과 부호 띠: 바닥과 꼭대기", "Curve coloured by the sign of f′ with a sign strip: bottoms and tops"],
      ["product rule 직사각형, chain rule 자 세 개", "Product-rule rectangle and three chain-rule rulers"],
      ["1D loss 곡선 위를 굴러가는 공 (공 끌기, learning rate slider)", "A ball rolling down a 1D loss curve (drag the ball, learning-rate slider)"],
      ["learning rate 네 개 비교, 고정 vs 점점 줄이기", "Four learning rates side by side; fixed vs decaying"],
      ["local minimum 시작점 지도, saddle point 등고선", "Start-point map for a local minimum; saddle-point contour map"],
      ["2D 등고선 위 gradient 화살표와 경로 (ascent 비교)", "Gradient arrows and the path on a 2D contour map (compare with ascent)"],
      ["층마다 곱해지는 gradient 막대 (vanishing, exploding)", "Gradient bars multiplied layer by layer (vanishing, exploding)"]
    ],
    udemy: ""
  },
  {
    n: 3, file: "ch03-neural-networks", ready: true,
    name: { ko: "ANN - I", en: "ANN - I" },
    topic: { ko: "perceptron에서 backprop까지", en: "From the perceptron to backprop" },
    goal: { ko: "neuron 하나 → 층 → 신경망. forward, loss, backprop 계산을 작은 network에서 숫자로 따라간다.", en: "One neuron → a layer → a network. Follow forward pass, loss and backprop with numbers on a tiny network." },
    sections: [
      ["인공 신경망이란, 생물 neuron과 비교", "What an ANN is, compared with biological neurons"],
      ["perceptron과 ANN 구조", "The perceptron and ANN architecture"],
      ["기하로 보는 ANN: 경계선 긋기", "A geometric view: drawing decision boundaries"],
      ["forward propagation: shape 따라가기", "Forward propagation: following the shapes"],
      ["error, loss, cost", "Errors, loss and cost"],
      ["backpropagation: chain rule을 층마다", "Backpropagation: the chain rule layer by layer"],
      ["작은 network를 손으로 한 번 학습하기", "Training a tiny network by hand, one step"],
      ["PyTorch로 같은 계산 (nn.Linear, autograd)", "The same in PyTorch (nn.Linear, autograd)"]
    ],
    figures: [
      ["생물 neuron과 인공 neuron 대응 그림", "Biological and artificial neurons side by side"],
      ["neuron 하나 계산기: 입력·weight·b slider, step vs sigmoid", "One-neuron calculator: input, weight and bias sliders; step vs sigmoid"],
      ["perceptron 경계선: w 끌기, perceptron 학습 규칙 animation, XOR에서 실패", "Perceptron boundary: drag w, animated perceptron learning rule, failure on XOR"],
      ["2-2-1 network forward 계산 한 단계씩 (행렬과 shape)", "2-2-1 forward pass step by step (matrices and shapes)"],
      ["BCE loss 곡선과 sample 4개의 cost", "BCE loss curves and the cost of four samples"],
      ["backprop 한 단계씩: δ가 거꾸로 흐른다", "Backprop step by step: δ flowing backwards"],
      ["browser에서 학습하는 2-h-1 network: 경계, 숨은 neuron의 선, loss 곡선", "A 2-h-1 network training in the browser: boundary, hidden neurons' lines, loss curve"]
    ],
    udemy: ""
  },
  {
    n: 4, file: "ch04-ann-practice", ready: true,
    name: { ko: "ANN - II", en: "ANN - II" },
    topic: { ko: "회귀, 분류, 깊이와 너비", en: "Regression, classification, depth and breadth" },
    goal: { ko: "회귀와 분류를 실제로 학습시키고, 층 수와 unit 수가 결과를 어떻게 바꾸는지 본다.", en: "Train regression and classification models, and see how the number of layers and units changes the result." },
    sections: [
      ["ANN으로 회귀 (기울기 바꿔 보기)", "Regression with an ANN (changing the slope)"],
      ["qwerties 분류: 2 class", "Classifying qwerties: two classes"],
      ["learning rate 비교", "Comparing learning rates"],
      ["multilayer와 비선형: 선형 층만 쌓으면 하나로 줄어든다", "Multilayer and nonlinearity: stacked linear layers collapse into one"],
      ["다중 출력: iris 3 class", "Multi-output: three iris classes"],
      ["hidden unit 수, depth vs breadth, parameter 수 세기", "Hidden units, depth vs breadth, counting parameters"],
      ["Sequential vs class로 model 정의", "Defining models with Sequential vs class"],
      ["DL model은 이해할 수 있나", "Are deep learning models understandable?"]
    ],
    figures: [
      ["회귀: 직선 하나를 학습, MSE와 r 비교 (기울기·잡음 slider)", "Regression: train one line, compare MSE and r (slope and noise sliders)"],
      ["qwerties: 거리 d를 바꾸면 다시 학습하는 경계와 정확도 곡선", "Qwerties: boundary retrained for each distance d, with the accuracy curve"],
      ["learning rate 실험 (PyTorch 28개 × seed 5개) + browser 한 번 학습", "Learning-rate experiment (PyTorch, 28 rates × 5 seeds) + one live browser run"],
      ["선형 층 두 개 = 행렬 하나 (W₂W₁ 계산 animation)", "Two linear layers = one matrix (animated W₂W₁)"],
      ["격자가 두 층을 지나며: activation 없음 / ReLU / tanh", "A grid passing through two layers: no activation / ReLU / tanh"],
      ["iris 3 class 경계와 누른 점의 softmax 막대", "Iris three-class boundary and softmax bars at a clicked point"],
      ["parameter 계산기, 깊이 × 너비 시험 정확도 격자 (spiral)", "Parameter counter and a depth × width test-accuracy grid (spiral)"],
      ["학습한 iris network의 weight 그림과 행렬", "Weights of the trained iris network as a diagram and matrices"]
    ],
    udemy: ""
  },
  {
    n: 5, file: "ch05-overfitting-regularization", ready: true,
    name: { ko: "Overfitting", en: "Overfitting" },
    topic: { ko: "overfitting, cross-validation, regularization", en: "Overfitting, cross-validation and regularization" },
    goal: { ko: "train 성적과 test 성적이 왜 갈라지는지 보고, data를 나누는 법(cross-validation)과 막는 법(dropout, L1/L2)을 익힌다.", en: "See why training and test scores drift apart, then learn how to split data (cross-validation) and how to prevent it (dropout, L1/L2)." },
    sections: [
      ["overfitting과 generalization", "Overfitting and generalization"],
      ["train, dev(validation), test 나누기", "Splitting into train, dev (validation) and test"],
      ["cross-validation (k-fold)", "Cross-validation (k-fold)"],
      ["scikit-learn과 DataLoader로 나누기, 회귀의 cross-validation", "Splitting with scikit-learn and DataLoader; cross-validation for regression"],
      ["regularization의 개념", "Regularization: the idea"],
      ["train()과 eval() mode", "train() and eval() modes"],
      ["dropout", "Dropout"],
      ["L1, L2 weight regularization", "L1 and L2 weight regularization"]
    ],
    figures: [
      ["다항식 차수 slider: 맞춘 곡선과 학습·시험 오차의 U자", "Polynomial degree slider: fitted curve and the U-shaped train/test errors"],
      ["data 100개를 train·dev·test로 나누는 그림", "100 samples split into train, dev and test"],
      ["k-fold 나누기 animation과 차수마다 검증 오차", "k-fold split animation and the validation error per degree"],
      ["ridge λ slider: 차수 14 곡선이 얌전해진다", "Ridge λ slider: the degree-14 curve calms down"],
      ["dropout: 학습 step마다 꺼지는 unit, dropout 있고 없이 나란히 학습", "Dropout: units switched off each step; training with and without dropout side by side"],
      ["L1 마름모 vs L2 원: 등고선과 해의 경로", "L1 diamond vs L2 circle: contours and solution paths"]
    ],
    udemy: ""
  },
  {
    n: 6, file: "ch06-activation-loss-normalization", ready: true,
    name: { ko: "Activation·Loss", en: "Activation & Loss" },
    topic: { ko: "activation function, loss function, normalization", en: "Activation functions, loss functions and normalization" },
    goal: { ko: "층 사이의 비선형(activation), 틀린 정도를 재는 방법(loss), 숫자 크기를 맞추는 방법(normalization)을 고르는 기준을 세운다.", en: "Learn how to choose the nonlinearity between layers (activation), the measure of error (loss), and how to put numbers on a common scale (normalization)." },
    sections: [
      ["activation function: sigmoid, tanh, ReLU 계열", "Activation functions: sigmoid, tanh, the ReLU family"],
      ["activation의 미분과 vanishing gradient", "Derivatives of activations and vanishing gradients"],
      ["loss function: MSE, BCE, cross-entropy", "Loss functions: MSE, BCE, cross-entropy"],
      ["출력층과 loss 짝 맞추기 (PyTorch 이름까지)", "Matching the output layer to the loss (with PyTorch names)"],
      ["data normalization: z-score, min-max", "Data normalization: z-score, min-max"],
      ["batch normalization", "Batch normalization"],
      ["예제 data: wine quality", "Worked dataset: wine quality"]
    ],
    figures: [
      ["activation 탐색기: 함수와 기울기를 나란히 (sigmoid, tanh, ReLU, LeakyReLU, ELU, GELU)", "Activation explorer: function and slope side by side (sigmoid, tanh, ReLU, LeakyReLU, ELU, GELU)"],
      ["숨은 층 10개 network의 층별 gradient 크기 (browser에서 계산)", "Gradient size per layer in a 10-hidden-layer network (computed in the browser)"],
      ["MSE vs cross-entropy: 같은 예측의 loss와 고치는 힘", "MSE vs cross-entropy: loss and push for the same prediction"],
      ["정규화 전후 loss 등고선과 GD 경로 (길쭉 → 둥글게)", "Loss contours and GD paths before and after normalization (stretched → round)"],
      ["batch normalization: batch마다 평균·분산 맞추기와 γ, β", "Batch normalization: per-batch mean and variance, then γ and β"],
      ["wine quality: 특성 크기와 정규화 전후 시험 정확도", "Wine quality: feature sizes and test accuracy before and after normalization"]
    ],
    udemy: ""
  },
  {
    n: 7, file: "ch07-optimizers", ready: true,
    name: { ko: "Optimizer", en: "Optimizer" },
    topic: { ko: "mini-batch, optimizer, learning rate", en: "Mini-batches, optimizers and learning rate" },
    goal: { ko: "gradient descent를 실제로 빠르고 안정되게 만드는 방법: mini-batch, momentum, Adam, learning rate schedule.", en: "What makes gradient descent fast and stable in practice: mini-batches, momentum, Adam, learning-rate schedules." },
    sections: [
      ["mini-batch 학습과 batch 크기", "Mini-batch training and batch size"],
      ["SGD와 momentum", "SGD and momentum"],
      ["RMSprop과 Adam", "RMSprop and Adam"],
      ["optimizer 비교", "Comparing optimizers"],
      ["Adam + L2 (weight decay)", "Adam with L2 (weight decay)"],
      ["learning rate decay와 schedule", "Learning-rate decay and schedules"],
      ["metaparameter 고르는 법", "How to pick metaparameters"]
    ],
    figures: [
      ["batch 크기에 따른 gradient 흔들림과 3 epoch 경로", "Gradient wobble by batch size, and the path over 3 epochs"],
      ["길쭉한 골짜기에서 SGD vs momentum (β slider)", "SGD vs momentum in a long valley (β slider)"],
      ["같은 골짜기에서 RMSprop, Adam", "RMSprop and Adam in the same valley"],
      ["굽은 골짜기 위 optimizer 경주 + moons network 학습 비교", "Optimizer race on a curved valley + training comparison on a moons network"],
      ["Adam + L2 vs AdamW: weight가 줄어드는 비율", "Adam + L2 vs AdamW: how fast weights shrink"],
      ["learning rate schedule 그래프 (계단, 지수, cosine, warmup)", "Learning-rate schedule plots (step, exponential, cosine, warmup)"],
      ["metaparameter 찾기: 격자 vs 무작위", "Searching metaparameters: grid vs random"]
    ],
    udemy: ""
  },
  {
    n: 8, file: "ch08-ffn-data-gpu", ready: true,
    name: { ko: "FFN", en: "FFN" },
    topic: { ko: "MNIST, data 다루기, GPU", en: "MNIST, handling data, GPUs" },
    goal: { ko: "손글씨 숫자 분류로 FFN 하나를 끝까지 만들고, data를 다루는 실전 문제(불균형, augmentation, 저장, GPU)를 푼다.", en: "Build one FFN end to end on handwritten digits, then handle real data problems: imbalance, augmentation, saving, GPUs." },
    sections: [
      ["fully-connected·feedforward network", "Fully-connected feedforward networks"],
      ["MNIST 숫자 분류 (binarize, normalize)", "Classifying MNIST digits (binarize, normalize)"],
      ["학습 전후 weight 분포", "Weight distributions before and after learning"],
      ["scrambled·shifted MNIST: FFN이 못 보는 것", "Scrambled and shifted MNIST: what an FFN cannot see"],
      ["universal approximation theorem", "The universal approximation theorem"],
      ["Dataset과 DataLoader의 구조, data 크기와 network 크기", "Anatomy of Dataset and DataLoader; data size vs network size"],
      ["unbalanced data, oversampling, noise·feature augmentation", "Unbalanced data, oversampling, noise and feature augmentation"],
      ["model 저장·불러오기, 가장 좋은 model 남기기, GPU 쓰기", "Saving and loading models, keeping the best one, using the GPU"]
    ],
    figures: [
      ["28×28 사진을 784칸 한 줄로 펼치는 animation", "Animation unrolling a 28×28 picture into a line of 784 cells"],
      ["숫자를 그리면 page 안의 MNIST network(196-64-10, 95.8%)가 확률을 낸다", "Draw a digit and the MNIST network inside the page (196-64-10, 95.8%) gives probabilities"],
      ["학습 전후 weight 분포와 숨은 neuron의 weight 무늬", "Weight distribution before and after training, and hidden neurons' weight patterns"],
      ["pixel 섞기, 옆으로 밀기와 정확도 곡선", "Scrambled pixels, sideways shifts and the accuracy curve"],
      ["universal approximation: sigmoid 계단 N개로 곡선 흉내", "Universal approximation: imitating a curve with N sigmoid steps"],
      ["data 수 × network 너비의 시험 정확도", "Test accuracy by data size × network width"],
      ["불균형 data와 oversampling, augmentation 미리보기", "Unbalanced data and oversampling; augmentation preview"],
      ["epoch마다 학습·dev 정확도와 가장 좋은 model", "Training and dev accuracy per epoch and the best model"]
    ],
    udemy: ""
  },
  {
    n: 9, file: "ch09-metrics-weights", ready: true,
    name: { ko: "Metrics·Init", en: "Metrics & Init" },
    topic: { ko: "성능 측정과 weight 초기화", en: "Measuring performance and weight initialization" },
    goal: { ko: "정확도 하나로는 안 보이는 것(precision, recall, F1)을 재고, 처음 weight를 어떻게 정해야 학습이 잘 되는지 본다.", en: "Measure what accuracy alone hides (precision, recall, F1), and see how the starting weights decide whether training works." },
    sections: [
      ["두 관점: confusion matrix", "Two perspectives: the confusion matrix"],
      ["accuracy, precision, recall, F1", "Accuracy, precision, recall, F1"],
      ["group 크기가 다를 때의 metric", "Metrics when group sizes differ"],
      ["계산 시간, test가 train보다 좋을 때", "Computation time; when test beats train"],
      ["weight matrix 크기 읽기", "Reading weight-matrix sizes"],
      ["weight 초기화가 중요한 이유", "Why weight initialization matters"],
      ["Xavier와 Kaiming 초기화", "Xavier and Kaiming initialization"],
      ["weight 얼리기(freeze)와 학습에 따른 weight 변화", "Freezing weights; how weights change during learning"],
      ["FFN 프로젝트 3개 (덧셈기, 심장병 예측, 빠진 값 채우기)", "Three FFN projects (adding machine, heart disease, missing-data interpolation)"]
    ],
    figures: [
      ["문턱 slider → 두 점수 분포와 confusion matrix", "Threshold slider → two score distributions and the confusion matrix"],
      ["문턱에 따른 accuracy, precision, recall, F1 곡선", "Accuracy, precision, recall and F1 curves against the threshold"],
      ["양성 비율에 따른 metric 막대 (드문 양성)", "Metric bars as positives become rare"],
      ["nn.Linear(4, 3).weight의 shape 읽기", "Reading the shape of nn.Linear(4, 3).weight"],
      ["같은 값 초기화 vs 무작위 초기화 (대칭), 초기화 크기에 따른 층별 activation histogram", "Equal vs random initialization (symmetry); per-layer activation histograms by init size"],
      ["Xavier vs Kaiming: 20층 동안 신호 크기", "Xavier vs Kaiming: signal size over 20 layers"],
      ["층별 weight 변화와 1층 얼리기", "Weight change per layer, and freezing layer 1"],
      ["FFN 프로젝트: 덧셈기 (browser 학습), 심장병, 빠진 값", "FFN projects: an adder (trained in the browser), heart disease, missing values"]
    ],
    udemy: ""
  },
  {
    n: 10, file: "ch10-autoencoders", ready: true,
    name: { ko: "Autoencoder", en: "Autoencoder" },
    topic: { ko: "autoencoder와 latent 공간", en: "Autoencoders and latent space" },
    goal: { ko: "압축했다가 다시 푸는 network로 잡음 제거, 가려진 부분 복원, latent 공간을 이해한다. 15장 diffusion으로 가는 다리.", en: "Use networks that compress and rebuild to remove noise, restore occlusions and understand latent space. A bridge to diffusion in chapter 15." },
    sections: [
      ["autoencoder: 압축하고 다시 풀기", "Autoencoders: compress, then rebuild"],
      ["denoising autoencoder", "Denoising autoencoders"],
      ["bottleneck에 unit 몇 개가 필요한가", "How many units the bottleneck needs"],
      ["가려진 부분 복원 (occlusion)", "Restoring occluded parts"],
      ["latent code와 latent 공간", "The latent code and latent space"],
      ["tied weights", "Tied weights"],
      ["생성 model로 가는 다리: latent에서 새 그림", "Bridge to generative models: new images from latent space"]
    ],
    figures: [
      ["원래 → 16개 code 막대 → 복원 (숫자 6개씩)", "Original → 16-number code bars → rebuilt (6 digits at a time)"],
      ["noise slider와 denoising 결과", "Noise slider and the denoised result"],
      ["bottleneck 크기 vs 복원 오차 곡선과 복원 모습", "Bottleneck size vs reconstruction error, with rebuilt digits"],
      ["가릴 줄을 고르는 slider와 복원 결과", "Sliders choosing which rows to hide, and the restored digit"],
      ["2D latent 공간 탐색기: 점을 누르면 decoder가 그린 숫자", "2D latent-space explorer: click a point, see the decoded digit"],
      ["tied weights 구조 그림과 MSE·parameter 비교", "Tied-weights diagram with MSE and parameter comparison"],
      ["latent 직선 보간과 무작위 점 decoding", "Straight-line interpolation in latent space and decoding random points"]
    ],
    udemy: ""
  },
  {
    n: 11, file: "ch11-cnn", ready: true,
    name: { ko: "CNN - I", en: "CNN - I" },
    topic: { ko: "convolution과 pooling", en: "Convolution and pooling" },
    goal: { ko: "kernel이 사진 위를 미끄러지며 특징을 찾는 방법과 CNN의 shape 계산을 손으로 따라간다.", en: "Follow by hand how a kernel slides over an image to find features, and how shapes flow through a CNN." },
    sections: [
      ["convolution의 개념: kernel이 미끄러진다", "Convolution: a kernel sliding over the image"],
      ["feature map과 kernel", "Feature maps and kernels"],
      ["stride, padding과 출력 크기 공식", "Stride, padding and the output-size formula"],
      ["transpose convolution", "Transpose convolution"],
      ["max·mean pooling, pool이냐 stride냐", "Max and mean pooling; pool or stride?"],
      ["표준 CNN 구조와 shape 따라가기", "The canonical CNN and following its shapes"],
      ["CNN으로 MNIST, Gaussian 분류, feature map 들여다보기", "Classifying MNIST and Gaussians; looking inside feature maps"],
      ["image transform과 custom DataLoader", "Image transforms and custom DataLoaders"],
      ["CNN 설계 연습: EMNIST, CIFAR10, FMNIST", "Design practice: EMNIST, CIFAR10, FMNIST"]
    ],
    figures: [
      ["convolution animation: 7×7 위를 미끄러지는 3×3 kernel과 칸별 곱셈", "Convolution animation: a 3×3 kernel sliding over 7×7, with the cell-by-cell products"],
      ["kernel 숫자를 바꿔 edge·blur 만들기 (실제 숫자에), 학습한 CNN의 kernel 8개와 feature map", "Edit kernel numbers to make edges or blur (on a real digit); the trained CNN's 8 kernels and feature maps"],
      ["stride·padding을 넣으면 출력 크기를 그리는 계산기", "Stride/padding calculator that draws the output size"],
      ["transpose convolution: 한 칸씩 kernel을 찍어 더하기", "Transpose convolution: stamping the kernel one cell at a time"],
      ["pooling 전후 비교 (max vs mean, 28 → 14 → 7)", "Before and after pooling (max vs mean, 28 → 14 → 7)"],
      ["CNN shape pipeline 그림 (층마다 (C, H, W)와 parameter)", "CNN shape pipeline with (C, H, W) and parameters at every layer"],
      ["숫자 그리기 → CNN 확률과 2층 feature map, 옮긴 숫자 정확도 CNN vs FFN, Gaussian 폭 예", "Draw a digit → CNN probabilities and layer-2 feature maps; shifted-digit accuracy CNN vs FFN; Gaussian width examples"],
      ["image transform 모음 (Rotation, Affine, scale, Flip, Erasing, Normalize)", "Image transform gallery (Rotation, Affine, scale, Flip, Erasing, Normalize)"],
      ["FMNIST CNN confusion matrix (칸을 누르면 수)", "FMNIST CNN confusion matrix (click a cell for its count)"]
    ],
    udemy: ""
  },
  {
    n: 12, file: "ch12-transfer-style", ready: true,
    name: { ko: "CNN - II", en: "CNN - II" },
    topic: { ko: "transfer learning과 style transfer", en: "Transfer learning and style transfer" },
    goal: { ko: "이미 학습된 network를 빌려 쓰는 방법과, feature의 통계(Gram matrix)로 그림 style을 옮기는 방법을 본다.", en: "Borrow networks that are already trained, and move a picture's style using feature statistics (the Gram matrix)." },
    sections: [
      ["transfer learning: 무엇을, 왜, 언제", "Transfer learning: what, why and when"],
      ["MNIST ↔ FMNIST: 작은 data로 transfer", "MNIST ↔ FMNIST: transfer with little data"],
      ["유명한 CNN 구조: AlexNet, VGG, ResNet", "Famous CNN architectures: AlexNet, VGG, ResNet"],
      ["ResNet-18, VGG-16으로 transfer (CIFAR10)", "Transfer with ResNet-18 and VGG-16 (CIFAR10)"],
      ["autoencoder로 pretraining", "Pretraining with autoencoders"],
      ["style transfer와 Gram matrix", "Style transfer and the Gram matrix"],
      ["style transfer algorithm", "The style transfer algorithm"]
    ],
    figures: [
      ["층별 freeze/unfreeze 스위치 그림 (MNIST ↔ FMNIST, 8가지 조합의 실제 정확도)", "Layer-by-layer freeze/unfreeze switches (MNIST ↔ FMNIST, measured accuracy of 8 combinations)"],
      ["residual connection: 더하기 길이 있을 때 gradient (30층, slider)", "Residual connections: gradients with a shortcut (30 layers, slider)"],
      ["CIFAR10 5,000장: 처음부터 vs ResNet-18·VGG-16 빌리기 막대", "CIFAR10 with 5,000 images: from scratch vs borrowed ResNet-18 and VGG-16 (bars)"],
      ["autoencoder pretraining 결과 막대", "Autoencoder pretraining results (bars)"],
      ["작은 feature map으로 Gram matrix 계산하기 (칸 누르기, 자리 섞기)", "Computing a Gram matrix from small feature maps (click cells, shuffle positions)"],
      ["style transfer 결과 (무늬 2가지, β 3단계)와 반복에 따른 변화", "Style-transfer results (2 styles, 3 values of β) and the change over iterations"]
    ],
    udemy: ""
  },
  {
    n: 13, file: "ch13-rnn", ready: true,
    name: { ko: "RNN", en: "RNN" },
    topic: { ko: "sequence, GRU, LSTM", en: "Sequences, GRU, LSTM" },
    goal: { ko: "순서가 있는 data를 hidden state로 이어 가며 처리하는 방법(RNN, GRU, LSTM)과 그 한계를 본다. 14장 attention의 출발점.", en: "Process ordered data by carrying a hidden state (RNN, GRU, LSTM) and see its limits. The starting point for attention in chapter 14." },
    sections: [
      ["sequence data가 따로 필요한 이유", "Why sequences need their own models"],
      ["RNN의 작동: hidden state가 이어진다", "How RNNs work: the hidden state carries over"],
      ["PyTorch의 RNN class와 shape", "The RNN class in PyTorch and its shapes"],
      ["교대 sequence 예측, sine wave 외삽", "Predicting alternating sequences; extrapolating a sine wave"],
      ["hidden state와 embedding", "Hidden states and embeddings"],
      ["GRU와 LSTM: gate로 기억 조절", "GRU and LSTM: gates that control memory"],
      ["RNN의 한계 → attention으로", "Limits of RNNs → on to attention"]
    ],
    figures: [
      ["순서 섞기: 값의 분포는 같고 신호만 사라진다", "Shuffling: same value distribution, the signal is gone"],
      ["펼친 RNN animation: time step마다 숫자 (입력을 눌러 바꾸기, w_x·w_h slider)", "Unrolled RNN animation with numbers at each time step (click inputs, w_x and w_h sliders)"],
      ["nn.RNN shape 계산기 (입력, 출력, h_n, parameter 수)", "nn.RNN shape calculator (input, output, h_n, parameter counts)"],
      ["sine wave 외삽: hidden 1·2·4·16", "Sine-wave extrapolation: hidden 1, 2, 4, 16"],
      ["글자 embedding 2D 지도와 browser 안 LSTM 글 만들기 (온도 slider)", "2-D map of character embeddings and LSTM text generation in the browser (temperature slider)"],
      ["LSTM gate slider: forget, input, output", "LSTM gate sliders: forget, input, output"],
      ["긴 sequence에서 gradient가 사라지는 그래프 (RNN, LSTM, forget bias 3)", "Gradients fading over a long sequence (RNN, LSTM, forget bias 3)"],
      ["첫 글자 기억하기 실험: 거리별 정확도", "Recall-the-first-bit experiment: accuracy by distance"]
    ],
    udemy: ""
  },
  {
    n: 14, file: "ch14-transformer", ready: true,
    name: { ko: "Transformer", en: "Transformer" },
    topic: { ko: "attention, ViT, LLM", en: "Attention, ViT, LLMs" },
    goal: { ko: "query, key, value로 '어디를 볼지' 정하는 attention에서 시작해 Transformer block, ViT, LLM까지 shape와 함께 따라간다.", en: "Start from attention, where query, key and value decide where to look, and follow the Transformer block, ViT and LLMs with their shapes." },
    sections: [
      ["attention의 아이디어: 어디를 볼까", "The idea of attention: where to look"],
      ["query, key, value와 scaled dot-product attention", "Query, key, value and scaled dot-product attention"],
      ["self-attention과 multi-head attention", "Self-attention and multi-head attention"],
      ["positional encoding", "Positional encoding"],
      ["Transformer block: attention + FFN + residual + layer norm", "The Transformer block: attention + FFN + residual + layer norm"],
      ["encoder, decoder, masking", "Encoder, decoder and masking"],
      ["Vision Transformer (ViT): 사진을 patch로", "Vision Transformer (ViT): images as patches"],
      ["LLM: 다음 token 예측", "LLMs: predicting the next token"]
    ],
    figures: [
      ["문장의 단어를 누르면 attention (GPT-2 small, 층·head slider)", "Click a word to see its attention (GPT-2 small, layer and head sliders)"],
      ["Q Kᵀ → softmax → × V를 shape와 함께 한 단계씩", "Q Kᵀ → softmax → × V, step by step with shapes"],
      ["head 하나의 attention 표 (바로 앞, it, 첫 token head)", "One head's attention table (previous-token, it, first-token heads)"],
      ["positional encoding 물결 그림과 자리 유사도 (사인 vs GPT-2 학습)", "Positional-encoding waves and position similarity (sinusoidal vs GPT-2 learned)"],
      ["Transformer block 그림과 parameter 표", "Transformer block diagram and parameter table"],
      ["causal mask 켜고 끄기", "Causal mask on and off"],
      ["ViT patch 나누기와 CLS attention", "ViT patch splitter and CLS attention"],
      ["GPT-2 token 나누기 (영어, 한국어)와 다음 token 확률 (온도 slider)", "GPT-2 tokenising (English, Korean) and next-token probabilities (temperature slider)"]
    ],
    udemy: "",
    note: { ko: "Udemy 강의에 없는 장. 'Attention·Transformer 따라가기' artifact의 내용을 바탕으로 새로 쓴다.", en: "Not in the Udemy course. Written new, building on the 'Attention·Transformer' artifact." }
  },
  {
    n: 15, file: "ch15-gan-diffusion", ready: true,
    name: { ko: "GAN·Diffusion", en: "GAN & Diffusion" },
    topic: { ko: "생성 model", en: "Generative models" },
    goal: { ko: "data의 분포에서 새 sample을 뽑는 두 방법: 경쟁으로 배우는 GAN과, noise를 더했다가 되돌리는 diffusion. 마지막으로 AI 윤리와 앞으로 공부하는 법.", en: "Two ways to draw new samples from a data distribution: GANs, which learn through competition, and diffusion, which adds noise and learns to undo it. Then AI ethics and how to keep learning." },
    sections: [
      ["생성 model이란: 분포에서 뽑기", "What generative models do: sampling from a distribution"],
      ["GAN: generator와 discriminator의 게임", "GANs: the generator–discriminator game"],
      ["linear GAN과 CNN GAN (MNIST, FMNIST, CIFAR)", "Linear and CNN GANs (MNIST, FMNIST, CIFAR)"],
      ["diffusion의 아이디어: noise를 더하고 되돌리기", "The idea of diffusion: add noise, then undo it"],
      ["forward process와 noise schedule", "The forward process and the noise schedule"],
      ["denoising network와 학습 (DDPM loss)", "The denoising network and its training (DDPM loss)"],
      ["sampling: noise에서 그림으로", "Sampling: from noise to an image"],
      ["condition, guidance, latent diffusion (Stable Diffusion)", "Conditioning, guidance and latent diffusion (Stable Diffusion)"],
      ["AI 윤리와 앞으로 공부하는 법 (논문 읽기)", "AI ethics and how to keep learning (reading papers)"]
    ],
    figures: [
      ["분포에서 뽑기: 키 분포와 새로 뽑은 20명", "Sampling a distribution: a height distribution and 20 new draws"],
      ["1D GAN: generator 분포가 data 분포를 따라가는 animation (봉우리 하나·둘, D(x) 곡선)", "1D GAN: the generator's distribution chasing the data's, animated (one or two peaks, D(x) curve)"],
      ["linear·CNN GAN이 만든 그림: MNIST, FMNIST, CIFAR10, epoch별", "Pictures from linear and CNN GANs: MNIST, FMNIST, CIFAR10, by epoch"],
      ["noise slider: 진짜 숫자에 noise를 더하는 forward process", "Noise slider: the forward process adding noise to a real digit"],
      ["2D 점 구름 diffusion: 퍼졌다가 되돌아오는 animation", "2D point-cloud diffusion: spreading out and coming back, animated"],
      ["noise schedule 그래프 (β_t, ᾱ_t, √ᾱ_t, √(1−ᾱ_t))", "Noise-schedule plots (β_t, ᾱ_t, √ᾱ_t, √(1−ᾱ_t))"],
      ["U-Net 구조 그림", "U-Net structure diagram"],
      ["DDPM sampling: noise에서 숫자까지 중간 모습", "DDPM sampling: snapshots from noise to digits"],
      ["classifier-free guidance w = 0, 1, 3으로 만든 숫자 0~9", "Digits 0–9 made with classifier-free guidance w = 0, 1, 3"]
    ],
    udemy: "",
    note: { ko: "diffusion 부분은 Udemy 강의에 없어 새로 쓴다.", en: "The diffusion part is not in the Udemy course and is written new." }
  }
];

if (typeof module !== "undefined") module.exports = { DLEDU_PLAN, DLEDU_STAGES };
