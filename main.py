import requests
from io import BytesIO
from flask import Flask, request, jsonify
from gradio_client import Client
from huggingface_hub import create_repo, upload_file

app = Flask(__name__)

@app.route('/run', methods=['POST'])
def run_model():
      # Obter parâmetros da consulta da URL
      endpoint = request.args.get('endpoint', default='https://pierroromeu-zbilatuca2testzz.hf.space')
      prompt = request.args.get('prompt', default='Hello!!')
      negative_prompt = request.args.get('negative_prompt', default='Hello!!')
      prompt_2 = request.args.get('prompt_2', default='Hello!!')
      negative_prompt_2 = request.args.get('negative_prompt_2', default='Hello!!')
      use_negative_prompt = request.args.get('use_negative_prompt', type=bool, default=True)
      use_prompt_2 = request.args.get('use_prompt_2', type=bool, default=True)
      use_negative_prompt_2 = request.args.get('use_negative_prompt_2', type=bool, default=False)
      seed = request.args.get('seed', type=int, default=0)
      width = request.args.get('width', type=int, default=256)
      height = request.args.get('height', type=int, default=256)
      guidance_scale = request.args.get('guidance_scale', type=float, default=5.5)
      num_inference_steps = request.args.get('num_inference_steps', type=int, default=50)
      strength = request.args.get('strength', type=float, default=0.7)
      use_vae_str = request.args.get('use_vae', default='false')  # Obtém use_vae como string
      use_vae = use_vae_str.lower() == 'true'  # Converte para booleano
      use_lora_str = request.args.get('use_lora', default='false')  # Obtém use_lora como string
      use_lora = use_lora_str.lower() == 'true'  # Converte para booleano
      use_img2img_str = request.args.get('use_img2img', default='false')  # Obtém use_vae como string
      use_img2img = use_img2img_str.lower() == 'true'  # Converte para booleano
      model = request.args.get('model', default='stabilityai/stable-diffusion-xl-base-1.0')
      vaecall = request.args.get('vaecall', default='madebyollin/sdxl-vae-fp16-fix')
      lora = request.args.get('lora', default='amazonaws-la/sdxl')
      lora_scale = request.args.get('lora_scale', type=float, default=0.7)
      url = request.args.get('url', default='https://example.com/image.png')

      # Chamar a API Gradio
      client = Client(endpoint)
      result = client.predict(
          prompt, negative_prompt, prompt_2, negative_prompt_2,
          use_negative_prompt, use_prompt_2, use_negative_prompt_2,
          seed, width, height,
          guidance_scale,
          num_inference_steps,
          strength,
          use_vae,
          use_lora,
          model,
          vaecall,
          lora,
          lora_scale,
          use_img2img,
          url,
          api_name="/run"
      )

      return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict_gan():
    # Obter parâmetros da consulta da URL
    endpoint = request.args.get('endpoint', default='https://pierroromeu-gfpgan.hf.space/--replicas/dgwcd/')
    hf_token = request.args.get('hf_token', default='')
    filepath = request.args.get('filepath', default='')
    version = request.args.get('version', default='v1.4')
    rescaling_factor = request.args.get('rescaling_factor', type=float, default=2.0)

    # Chamar a API Gradio
    client = Client(endpoint, hf_token=hf_token)
    result = client.predict(
        filepath,
        version,
        rescaling_factor,
        api_name="/predict"
    )

    return jsonify(result)

@app.route('/faceswapper', methods=['POST'])
def faceswapper():
    # Obter parâmetros da consulta da URL
    endpoint = request.args.get('endpoint', default='https://pierroromeu-faceswapper.hf.space/--replicas/u42x7/')
    user_photo = request.args.get('user_photo', default='')
    result_photo = request.args.get('result_photo', default='')
  
    # Chamar a API Gradio
    client = Client(endpoint)
    result = client.predict(
        user_photo,
        result_photo,
        api_name="/predict"
    )

    return jsonify(result)

@app.route('/train', methods=['POST'])
def answer():
    # Obter parâmetros da consulta da URL
    token = request.args.get('token', default='')
    endpoint = request.args.get('endpoint', default='https://pierroromeu-gfpgan.hf.space/--replicas/dgwcd/')
    dataset_id=request.args.get('dataset_id', default='')
    output_model_folder_name=request.args.get('output_model_folder_name', default='')
    concept_prompt=request.args.get('concept_prompt', default='')
    max_training_steps=request.args.get('max_training_steps', type=int, default=0)
    checkpoints_steps=request.args.get('checkpoints_steps', type=int, default=0)
    remove_gpu_after_training_str = request.args.get('remove_gpu_after_training', default='false')  # Obtém como string
    remove_gpu_after_training = remove_gpu_after_training_str.lower() == 'true'  # Converte para booleano

    # Chamar a API Gradio
    client = Client(endpoint, hf_token=token)
    result = client.predict(
        dataset_id,
        output_model_folder_name,
        concept_prompt,
        max_training_steps,
        checkpoints_steps,
        remove_gpu_after_training,
        api_name="/main"
    )

    return jsonify(result)

@app.route('/verify', methods=['GET'])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return jsonify('Check')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    # Parâmetros
    file_name= request.args.get('file_name', default='')
    repo = request.args.get('repo', default='')
    url = request.args.get('url', default='')
    token = request.args.get('token', default='')

    try:
        # Crie o repositório
        repo_id = repo
        create_repo(repo_id=repo_id, token=token)

        # Faça o download do conteúdo da URL em memória
        response = requests.get(url)
        if response.status_code == 200:
            # Obtenha o conteúdo do arquivo em bytes
            file_content = response.content
            # Crie um objeto de arquivo em memória
            file_obj = BytesIO(file_content)
            # Faça o upload do arquivo
            upload_file(
                path_or_fileobj=file_obj,
                path_in_repo=file_name,
                repo_id=repo_id,
                token=token
            )

            # Mensagem de sucesso
            return jsonify({"message": "Sucess"})
        else:
            return jsonify({"error": "Failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
