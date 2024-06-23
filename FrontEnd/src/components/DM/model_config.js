import ort from 'onnxruntime-web/webgpu';


function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter()
        return adapter.features.has('shader-f16')
    } catch (e) {
        return false
    }
}

document.addEventListener("DOMContentLoaded", () => {
    hasFp16().then((fp16) => {
        if (fp16) {
            loading = load_models(models);
        } else {
            log("Your GPU or Browser doesn't support webgpu/f16");
        }
    });
});


/*
 * load models used in the pipeline
 */
async function load_models(models) {
    const cache = await caches.open("onnx");
    let missing = 0;
    for (const [name, model] of Object.entries(models)) {
        const url = `${config.model}/${model.url}`;
        let cachedResponse = await cache.match(url);
        if (cachedResponse === undefined) {
            missing += model.size;
        }
    }
    if (missing > 0) {
        log(`downloading ${missing} MB from network ... it might take a while`);
    } else {
        log("loading...");
    }
    for (const [name, model] of Object.entries(models)) {
        try {
            const start = performance.now();
            const model_bytes = await fetchAndCache(config.model, model.url);
            const sess_opt = { ...opt, ...model.opt };
            models[name].sess = await ort.InferenceSession.create(model_bytes, sess_opt);
            const stop = performance.now();
            log(`${model.url} in ${(stop - start).toFixed(1)}ms`);
        } catch (e) {
            log(`${model.url} failed, ${e}`);
        }
    }
    log("ready.");
}
