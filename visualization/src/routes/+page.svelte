<script>
  import LabelTransitionFlow from '$lib/components/LabelTransitionFlow.svelte';
  import IterationLabelGrid from '$lib/components/IterationLabelGrid.svelte';
  import TextComparisonViewer from '$lib/components/TextComparisonViewer.svelte';
  import ScholaWriteDetailPanel from '$lib/components/ScholaWriteDetailPanel.svelte';
    import SemanticViews from '$lib/components/SemanticViews.svelte';


  function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

</script>

<nav class="top-nav">
  <div class="nav-title">ScholaWrite Dashboard</div>
  <ul>
    <li><a href="#schola">ScholaWrite Dataset</a></li>
    <li><a href="#llm">LLM Experiments</a></li>
    <li><a href="#compare">LLM vs Human</a></li>
  </ul>
</nav>

<h1 class="dashboard-title">📘 ScholaWrite Visualization System</h1>

<!-- Section 1 -->
<section id="schola" class="main-block block-schola">
  <h2>ScholaWrite Dataset Visualization</h2>
  <p class="block-desc">Visualizing how real human writing labels evolve over time.</p>


  <div class="sub-section">
    <h3>ScholaWrite Dataset Detail Panel</h3>
    <ScholaWriteDetailPanel />
  </div>

  <div class="sub-section">
    <h3>Label Transition Flow</h3>
    <LabelTransitionFlow />
  </div>
</section>

<!-- Section 2 -->
<section id="llm" class="main-block block-llm">
  <h2>LLM Generation Evolution</h2>
  <p class="block-desc">Tracking the iteration process and label dynamics of LLM outputs.</p>

  <div class="sub-section">
    <h3>Iteration Label Grid</h3>
    <IterationLabelGrid />
  </div>

  <div class="sub-section">
    <h3>Text Comparison and Highlighting</h3>
    <TextComparisonViewer />
  </div>
</section>


<section id="compare" class="main-block block-compare">
  <h2>Semantic Visualization</h2>
  <p class="block-desc">Analyzing differences in semantics during writing iteration.</p>

  <div class="sub-section">
    <SemanticViews />
  </div>
</section>

<!-- Section 3 -->
<section id="compare" class="main-block block-compare">
  <h2>LLM vs Human Revision</h2>
  <p class="block-desc">Analyzing differences in revision styles and strategies.</p>

  <div class="sub-section">
    <h3>(Coming Soon)</h3>
    <p>Stay tuned...</p>
  </div>
</section>

<button class="scroll-top-btn" on:click={scrollToTop}>↑ Top</button>


<style>
  .top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #1c2833;
    color: white;
    padding: 0.8rem 2rem;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .nav-title {
    font-size: 1.2rem;
    font-weight: bold;
  }

  .top-nav ul {
    list-style: none;
    display: flex;
    gap: 1.5rem;
    margin: 0;
    padding: 0;
  }

  .top-nav li a {
    color: white;
    text-decoration: none;
    font-weight: 500;
  }

  .top-nav li a:hover {
    text-decoration: underline;
  }

  .dashboard-title {
    text-align: center;
    font-size: 2.2rem;
    margin: 2rem 0;
    color: #1c2833;
  }

  .main-block {
    padding: 3rem 2rem;
    border-radius: 16px;
    margin: 3rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }

  .block-desc {
    color: #555;
    font-size: 1rem;
    margin-bottom: 2rem;
  }

  .sub-section {
    margin-top: 2rem;
    padding: 1.5rem;
    background: #f7f9fa;
    border-radius: 12px;
    border: 1px solid #ddd;
  }

  .block-schola {
    background: #e8f4fd;
    border-left: 6px solid #3498db;
  }

  .block-llm {
    background: #fff4e6;
    border-left: 6px solid #f39c12;
  }

  .block-compare {
    background: #f0f0f0;
    border-left: 6px solid #7f8c8d;
  }

  h2 {
    margin-top: 0;
    font-size: 1.5rem;
    color: #2c3e50;
  }

  h3 {
    margin-bottom: 0.5rem;
    font-size: 1.2rem;
    color: #34495e;
  }

  .scroll-top-btn {
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 0.6rem 1rem;
  background: #4e79a7;
  color: white;
  border: none;
  border-radius: 50%;
  font-size: 1.2rem;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
  z-index: 999;
  transition: background-color 0.2s ease;
}

.scroll-top-btn:hover {
  background-color: #3a5f8a;
}

</style>
