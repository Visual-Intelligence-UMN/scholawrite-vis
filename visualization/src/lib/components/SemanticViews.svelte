<script lang="ts">
    import * as d3 from 'd3';
    import * as tf from '@tensorflow/tfjs';
    import * as use from '@tensorflow-models/universal-sentence-encoder';
    import { onMount } from 'svelte';
    

    let textData: string[] = [];
    let embeddings: any[] = [];

    let llmState: string = 'gpt'

    function cosineSimilarity(a: tf.Tensor1D | number[], b: tf.Tensor1D | number[]): number {
        const aTensor = tf.tensor1d(a);
        const bTensor = tf.tensor1d(b);

        const dotProduct = tf.dot(aTensor, bTensor); 
        const normA = tf.norm(aTensor);
        const normB = tf.norm(bTensor);

        const similarity = dotProduct.div(normA.mul(normB));

        return similarity.dataSync()[0];
    }

    async function computeSimilarities(llmState: string){
        const model = await use.load();

        const dataPathJSON = await fetch(`/data/text-comparison/${llmState}/files.json`);
        const dataPaths = await dataPathJSON.json();

        console.log('Data Paths:', dataPaths);
        
        // Example text data
        textData = await Promise.all(
            dataPaths.map(async (path: string) => {
                const response = await fetch(`/data/text-comparison/${llmState}/${path}`);
                return await response.text();
            })
        )

        // Generate embeddings for each text
        embeddings = await Promise.all(textData.map(async (text) => {
            const embedding = await model.embed(text);
            return embedding.arraySync()[0];
        }));

        console.log('Embeddings:', embeddings);

        // compute adjacent cosine similarities
        const adjacentSimilarities = [];
        for (let i = 0; i < embeddings.length-1; i++) {
            const sim = cosineSimilarity(embeddings[i], embeddings[i+1]);
            adjacentSimilarities.push({ index:i, similarity: sim });
        }

        console.log('Adjacent Cosine Similarities:', adjacentSimilarities);

        // compute cosine similarities with a reference text
        const referenceSimilarities = [];
        for (let i=0; i< embeddings.length; i++) {
            const sim = cosineSimilarity(embeddings[i], embeddings[0]); 
            referenceSimilarities.push({ index: i, similarity: sim });
        }

        console.log('Cosine Similarities with Reference:', referenceSimilarities);

        return referenceSimilarities;
    }

    function renderLinePlot(referenceSimilarities: { index: number; similarity: number }[]) {
        // create data visualization
        const svgWidth = 800;
        const svgHeight = 400;
        const margin = { top: 20, right: 30, bottom: 30, left: 40 };

        const width = svgWidth - margin.left - margin.right;
        const height = svgHeight - margin.top - margin.bottom;

        // Remove previous SVG if any
        d3.select('#similarity-plot').selectAll('*').remove();

        // Create SVG
        const svg = d3
            .select('#similarity-plot')
            .append('svg')
            .attr('width', svgWidth)
            .attr('height', svgHeight)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // X and Y scales
        const x = d3
            .scaleLinear()
            .domain([0, referenceSimilarities.length - 1])
            .range([0, width]);

        const y = d3
            .scaleLinear()
            .domain([0, 1])
            .range([height, 0]);

        // X and Y axes
        svg
            .append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).tickValues(
                    d3.range(0, referenceSimilarities.length, 10)
                ));

        svg.append('g').call(d3.axisLeft(y));

        // Line generator
        const line = d3
            .line<{ index: number; similarity: number }>()
            .x((d) => x(d.index))
            .y((d) => y(d.similarity));

        // Draw the line
        svg
            .append('path')
            .datum(referenceSimilarities)
            .attr('fill', 'none')
            .attr('stroke', 'steelblue')
            .attr('stroke-width', 2)
            .attr('d', line);

        // === Hover UI elements ===
        const focusLine = svg.append('line')
        .attr('stroke', 'gray')
        .attr('stroke-width', 1)
        .attr('y1', 0)
        .attr('y2', height)
        .style('display', 'none');

        const focusCircle = svg.append('circle')
        .attr('r', 4)
        .attr('fill', 'red')
        .style('display', 'none');

        const focusText = svg.append('text')
        .attr('fill', 'black')
        .attr('font-size', '12px')
        .attr('x', 10)
        .attr('y', 10)
        .style('display', 'none');

        // Overlay to capture mouse
        svg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', 'none')
        .attr('pointer-events', 'all')
        .on('mousemove', function (event) {
            const [mx] = d3.pointer(event);
            const i = Math.round(x.invert(mx));
            if (i >= 0 && i < referenceSimilarities.length) {
            const d = referenceSimilarities[i];
            const cx = x(d.index);
            const cy = y(d.similarity);

            focusLine
                .attr('x1', cx)
                .attr('x2', cx)
                .style('display', null);

            focusCircle
                .attr('cx', cx)
                .attr('cy', cy)
                .style('display', null);

            focusText
                .attr('x', cx + 10)
                .attr('y', cy - 10)
                .text(`Iter ${d.index}, Sim ${d.similarity.toFixed(2)}`)
                .style('display', null);
            }
        })
        .on('mouseout', () => {
            focusLine.style('display', 'none');
            focusCircle.style('display', 'none');
            focusText.style('display', 'none');
        });


    }

    onMount(async () => {
        // Load the Universal Sentence Encoder model
        const referenceSimilarities = await computeSimilarities(llmState);
        await renderLinePlot(referenceSimilarities);
    });

    $: if (llmState) {
        computeSimilarities(llmState).then(renderLinePlot);
    }


</script>

<div>
    <select id='llm-state' bind:value={llmState}>
        <option value='gpt'>GPT</option>
        <option value='gpt4.1-mini'>GPT 4.1 mini</option>
        <option value='llama_meta'>llama meta</option>
        <option value='llama_sw'>Llama SW</option>
    </select>
    <div id="similarity-plot"></div>
</div>