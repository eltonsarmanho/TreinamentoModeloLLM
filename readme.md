# Passos para Instalar e Configurar CUDA para Treinamento com PyTorch

1. **Instalar o driver NVIDIA compatível**
   - Baixe e instale o driver mais recente para sua GPU NVIDIA.
   - Verifique a instalação com:
     ```bash
     nvidia-smi
     ```
   - O comando deve mostrar informações da sua GPU e versão do driver.

2. **Instalar o CUDA Toolkit**
   - Baixe o CUDA Toolkit compatível com seu driver no site da NVIDIA.
   - Instale seguindo as instruções do site.
   - Verifique a instalação com:
     ```bash
     nvcc --version
     ```

3. **Configurar variáveis de ambiente**
   - Adicione ao seu `.bashrc` ou execute no terminal:
     ```bash
     export PATH=/usr/local/cuda/bin:$PATH
     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
     ```

4. **Instalar PyTorch com suporte CUDA**
   - No ambiente virtual, execute:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - Ajuste `cu121` para a versão CUDA instalada (exemplo: `cu120` para CUDA 12.0).

5. **Instalar Hugging Face Transformers e Datasets**
   ```bash
   pip install transformers datasets
   ```

6. **Verificar se PyTorch reconhece a GPU**
   - No Python:
     ```python
     import torch
     print(torch.cuda.is_available())  # Deve retornar True
     ```

7. **(Opcional) Instalar cuDNN**
   - Para algumas aplicações, instale:
     ```bash
     pip install nvidia-cudnn-cu12
     ```

8. **Reinicie o terminal ou ambiente virtual após as instalações.**

Pronto! Seu ambiente estará preparado para treinar modelos na GPU usando