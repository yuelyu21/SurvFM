# Shared regression interface

For training fold 𝒟<sub>train</sub> and horizon τ, define the context table

<div class="math-display" role="math" aria-label="C tau of D train is the set of pairs X i and clipped Z i tau for training indices.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>𝒞</mi><mi>τ</mi></msub><mo>(</mo><msub><mi>𝒟</mi><mi>train</mi></msub><mo>)</mo>
  <mo>=</mo><mo>{</mo><mo>(</mo><msub><mi>X</mi><mi>i</mi></msub><mo>,</mo>
  <msub><mover><mi>Z</mi><mo>~</mo></mover><mi>i</mi></msub><mo>(</mo><mi>τ</mi><mo>)</mo><mo>)</mo>
  <mo>:</mo><mi>i</mi><mo>∈</mo><msub><mi>I</mi><mi>train</mi></msub><mo>}</mo><mo>.</mo>
</math>
</div>

Every backbone receives this same supervised regression problem. Its empirical
squared-error representation is

<div class="math-display" role="math" aria-label="L tau b of f is one over the number of training observations times the sum of squared clipped pseudo target minus f of X i.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>ℒ</mi><mrow><mi>τ</mi><mo>,</mo><mi>b</mi></mrow></msub><mo>(</mo><mi>f</mi><mo>)</mo>
  <mo>=</mo><mfrac><mn>1</mn><mrow><mo>|</mo><msub><mi>I</mi><mi>train</mi></msub><mo>|</mo></mrow></mfrac>
  <munder><mo>∑</mo><mrow><mi>i</mi><mo>∈</mo><msub><mi>I</mi><mi>train</mi></msub></mrow></munder>
  <msup><mrow><mo>{</mo><msub><mover><mi>Z</mi><mo>~</mo></mover><mi>i</mi></msub><mo>(</mo><mi>τ</mi><mo>)</mo><mo>−</mo><mi>f</mi><mo>(</mo><msub><mi>X</mi><mi>i</mi></msub><mo>)</mo><mo>}</mo></mrow><mn>2</mn></msup><mo>.</mo>
</math>
</div>

This equation defines the common task; it does not claim that every backbone
updates task-specific weights. In-context backbones condition on the supplied
training table, whereas conventional regressors fit dataset-specific
parameters.

For backbone 𝒜<sub>*b*</sub> and held-out observation *j*,

<div class="math-display" role="math" aria-label="Predicted m tau b of X j is backbone A b applied to the context table and X j.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mover><mi>m</mi><mo>^</mo></mover><mrow><mi>τ</mi><mo>,</mo><mi>b</mi></mrow></msub><mo>(</mo><msub><mi>X</mi><mi>j</mi></msub><mo>)</mo>
  <mo>=</mo><msub><mi>𝒜</mi><mi>b</mi></msub><mo>{</mo>
  <msub><mi>𝒞</mi><mi>τ</mi></msub><mo>(</mo><msub><mi>𝒟</mi><mi>train</mi></msub><mo>)</mo><mo>,</mo><msub><mi>X</mi><mi>j</mi></msub><mo>}</mo><mo>.</mo>
</math>
</div>

Predicted RMST is larger for longer restricted event-free time. Concordance
metrics require the risk orientation

<div class="math-display" role="math" aria-label="Predicted risk equals negative bounded predicted RMST.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mover><mi>r</mi><mo>^</mo></mover><mrow><mi>τ</mi><mo>,</mo><mi>b</mi></mrow></msub><mo>(</mo><msub><mi>X</mi><mi>j</mi></msub><mo>)</mo>
  <mo>=</mo><mo>−</mo><msub><mover><mi>m</mi><mo>~</mo></mover><mrow><mi>τ</mi><mo>,</mo><mi>b</mi></mrow></msub><mo>(</mo><msub><mi>X</mi><mi>j</mi></msub><mo>)</mo><mo>.</mo>
</math>
</div>

The package exposes these two orientations directly as `predict()` and
`predict_risk()`.
