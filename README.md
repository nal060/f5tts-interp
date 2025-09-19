# f5tts-interp

interpretability research on f5-tts focusing on understanding what happens inside the mmdit transformer blocks

## main files you actually need

`scripts/extract_block_outputs.py` - hooks into the mmdit model and grabs activations from each transformer block and ffn layer, this is how we see whats happening inside

`scripts/inject_block_outputs.py` - takes those extracted activations and shoves them back into the model, lets you test what happens when you mess with specific layers

`scripts/end_to_end_pipeline.py` - the full research pipeline that combines extraction and injection with a clean class interface and some experiments just to see how it works.

`scripts/test_pipeline.py` - makes sure everything works by checking that extract then inject unchanged gives you the exact same output as normal forward pass

## other stuff

`scripts/toy_model_*.py` - simple examples to understand how hooks and patching work without the complexity of the full model

`model/` - copy of f5-tts model code with extra comments explaining everything

right now we can handle block outputs and ffn layers, later well dig into attention matrices and pre-projection outputs but this is enough to start doing real research
