Forward Pass
hdx1  hdxhd   hdx1     hdxF   Fx1   hdx1    <---- Matrix Dimensions
z_t = W_hh @ h_{t-1} + W_xh @ x_t + b_h 
h_t = tanh(z_t)

odx1    od*hd   hdx1   odx1                 <---- Matrix Dimensions
o_t  =  W_hy  @  h_t + b_y
y_t = softmax(o_t) = exp(o_t) / Σⱼ exp(o_t[j])

-------------------------------------------------------------------

Compute Loss
L_t = -log(y_t[c])    <---- cross entopy loss.
Actually L_t = -Σᵢ target_t[i] × log(y_t[i]) , target_t[i] = 0 (all other classes except correct class c) which simplifies to above equation

-------------------------------------------------------------------

Back Prop => ∂y_t[i] has odx1               <---- Matrix Dimensions
∂L_t/∂y_t[i] = {  -1/y_t[c]   if i = c (correct class)
────────────   {   0          if i ≠ c (other classes)

∂L_t/∂o_t = ∂L_t/∂y_t * ∂y_t/∂o_t = y_t - one_hot(c)

∂y_t/∂o_t? This simplifies to below
∂L_t/∂o_t[i] = {  y_t[i] - 1   if i = c (correct class)
────────────   {  y_t[i]       if i ≠ c (other classes)

∂L_t/∂W_hy[i] = ∂L_t/∂y_t * ∂y_t/∂o_t * ∂o_t/∂W_hy, ∂L_t/∂by[i] = ∂L_t/∂y_t * ∂y_t/∂o_t * ∂o_t/∂by = ∂L_t/∂y_t * ∂y_t/∂o_t
∂o_t/∂W_hy =    h_t.T, 
∂o_t/∂by = 1
**∂L_t/∂W_hy** = (y_t - one_hot(c)) * h_t.T
**∂L_t/∂by** = (y_t - one_hot(c))
────────────

∂L_t/∂h_t = ∂L_t/∂y_t * ∂y_t/∂o_t * ∂o_t/∂h_t = ∂L_t/∂y_t * ∂y_t/∂o_t * W_hy.T
∂L_t/∂h_t = (y_t - one_hot(c)) * ∂o_t/∂h_t = (y_t - one_hot(c)) * W_hy.T
∂o_t/∂h_t =  W_hy.T

∂L_t/∂z_t = ∂L_t/∂h_t * ∂h_t/∂z_t
∂h_t/∂z_t = derivative of tanh(z_t) = 1 - tanh^2(z_t) = 1 - h_t^2
∂L_t/∂z_t = ∂L_t/∂h_t * 1 - h_t^2
∂L_t/∂z_t = (y_t - one_hot(c)) * W_hy.T  * 1 - h_t^2

∂L_t/∂W_hh =  ∂L_t/∂z_t * ∂z_t/∂W_hh
∂z_t/∂W_hh = h_{t-1}.T
∂z_t/∂W_xh = x_t.T
∂z_t/∂bh   = 1
∂L_t/∂W_hh =  ∂L_t/∂z_t * h_{t-1}.T, ∂L_t/∂W_xh =  ∂L_t/∂z_t * x_t.T, ∂L_t/∂bh =  ∂L_t/∂z_t
**∂L_t/∂W_hh** =  (y_t - one_hot(c)) * W_hy.T  * 1 - h_t^2 * h_{t-1}.T, 
**∂L_t/∂W_xh** =  (y_t - one_hot(c)) * W_hy.T  * 1 - h_t^2 * x_t.T, 
**∂L_t/∂bh** =  (y_t - one_hot(c)) * W_hy.T  * 1 - h_t^2
────────────
