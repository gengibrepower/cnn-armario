import torch

print("Versão do PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    print("Número de GPUs:", torch.cuda.device_count())
    
    # Teste simples de operação na GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("Operação na GPU realizada com sucesso!")
else:
    print("⚠️ GPU NÃO DETECTADA — PyTorch está rodando na CPU.")
