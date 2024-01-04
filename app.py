import gradio as gr
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

def adjust_demo_process(input_img):
    global model, task_prompt
    pixel_values = processor(input_img, return_tensors="pt").pixel_values
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

    # set model device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        
        # modify parameters
        early_stopping=True,
        num_beams=2,
        output_scores=True,
    )
    
    # data post-processing
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    print(sequence)

    # convert sequence to json
    output = processor.token2json(sequence)
    # output = pd.DataFrame.from_dict(output)
    return output

model.eval()

demo = gr.Interface(
    fn=adjust_demo_process,
    inputs="image",
    outputs="json",
    title=f"🗃️ Streamlining Invoice Processing: Document Understanding Transformer",
)
demo.launch(debug=True, share=True)