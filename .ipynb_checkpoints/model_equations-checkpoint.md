\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\begin{document}


$$Acne\,System\,State\,Vector/Treatment\,Input\,Vector: 
v_{t} = \begin{pmatrix}
B_t \\
I_t \\
S_t
\end{pmatrix}, u_{t} = \begin{pmatrix}days_{antibiotics}\\cream_{used}\\T(tstd)\end{pmatrix}\\\\v_{t+1} = F_{\theta}(v_{t-1}, u_{t}\mid \theta)\\\\\
Severity\,State\,As\,A\,Function\,of\,v_{t}: 
x_{t}= C \cdot v_{t} + m_{t-1}$$


$$B_t: Bacterial\,Facial\,Load - \\
B_t = B_{t-1} + r_{growth} \cdot B_{t-1}\frac{1-B_{t-1}}{K_{CC}}-k_{antibiotics} \cdot days_{antibiotics} \cdot B_{t-1} + k_{sebum} \cdot B_{t-1} \cdot S_{t-1} + noise$$

$$I_t: Inflammatory\,Activity - \\
I_t = I_{t-1} + I_{bacterial \, induction} \cdot B_{t-1} - I_{decay}/T(tstd)\cdot T(tstd) - I_{baseline decay} \cdot I_{t-1} + noise\\\\S_b: Sebum\,State - \\
S_t = S_{t-1} + r_{I production} \cdot I_{t-1} -r_{cream \, clean} \cdot cream_{used} + noise$$

$$\\\\Acne\,System\,State\,Inference\,Step-\\ \{v_{t}\} = EKF(v_{t}\mid \theta)\\\\EM\,Step - \\$$  
$$\theta_{k+1} = argmax(\mathop{\mathbb{E}_{v(t),\theta_{k}\mid}}[logP(x, v(t)\mid\theta_{k})])$$





