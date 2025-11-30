import torch
from torch import nn

class PatchEmbed(nn.Module):
  def __init__(self,img_size, patch_size, in_ch, embed_dims) -> None:
    super().__init__()
    self.patch_size = patch_size
    self.grid_size = img_size // patch_size
    self.num_patches = self.grid_size ** 2
    self.proj_layer = nn.Conv2d(in_channels=in_ch,
                                out_channels=embed_dims,
                                kernel_size=patch_size,
                                stride=patch_size)
    
  def forward(self, x: torch.Tensor):
    x = self.proj_layer(x)
    # выпрямляем 16 получившихся патчей 7х7
    # и получаем 16 векторов, но не 49 измерений
    # а embed_dims измерений
    x = x.flatten(2).transpose(1, 2)
    return x
  
  
class Attention(nn.Module):
  def __init__(self, dim, num_heads) -> None:
    super().__init__()
    self.num_heads = num_heads
    self.qkv = nn.Linear(dim, dim*3)
    self.proj_layer = nn.Linear(dim, dim)
    
  def forward(self, x: torch.Tensor):
    B, N, C = x.shape
    # разбиваем каждый вектор на сумму трёх
    # решейпом каждый вектор на 3 делим
    # чтобы запихнуть в q, k и v
    # а также на кол-во голов делим
    # ибо q, k и v есть в каждой из голов
    qkv: torch.Tensor = self.qkv(x)
    qkv = qkv.view(B, N, 3, self.num_heads, C // self.num_heads)
    # меняем порядок осей
    # получаем [B, H, N, C_head]
    q, k, v = qkv.permute(2,0,3,1,4)
    # знаменатель из формулы внимания
    scale = (C // self.num_heads) ** -0.5
    # само внимание
    attn: torch.Tensor = (q @ k.transpose(-2, -1)) * scale
    # активация (получаем вероятности обращения внимания)
    attn = attn.softmax(-1)
    # применение внимания
    # до этого разъединяли, теперь
    # объединяем все данные со всех голов
    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    out = self.proj_layer(out)
    return out
  
class FeedForward(nn.Module):
  def __init__(self, dim, hid) -> None:
    super().__init__()
    self.fc1 = nn.Linear(dim, hid)
    self.fc2 = nn.Linear(hid, dim)
    self.dropout = nn.Dropout(0.1)
    self.act = nn.GELU()
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x
  
class EncoderBlock(nn.Module):
  def __init__(self, dim, num_heads, feed_ratio) -> None:
    super().__init__()
    self.norm1 = nn.LayerNorm(dim)
    self.attn = Attention(dim, num_heads)
    self.norm2 = nn.LayerNorm(dim)
    # насколько увеличим глубину векторов
    self.feed_fwd = FeedForward(dim, int(dim*feed_ratio))
    
  def forward(self, x):
    # ресидуал на до и после аттеншна
    x = x + self.attn(self.norm1(x))
    # ресидуал на до и после фид-форварда
    x = x + self.feed_fwd(self.norm2(x))
    return x
  
class Encoder(nn.Module):
  def __init__(self, img_size, patch_size, in_ch, embed_dims, num_heads) -> None:
    super().__init__()
    self.patch_embed = PatchEmbed(img_size=img_size,
                                  patch_size=patch_size,
                                  in_ch=in_ch,
                                  embed_dims=embed_dims)
    num_patches = self.patch_embed.num_patches
    # случайный обучаемый токен на который наращивается инфа о классе фотки
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
    # инфа о положении патчей
    # num_patches + 1 потому чуто токен класса приклеиваем
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dims))
    self.attn_block = EncoderBlock(dim=embed_dims, num_heads=2,
                                     feed_ratio=2)
    self.norm = nn.LayerNorm(embed_dims)
    # Инициализируем обучаемые позиции и класс токен
    nn.init.trunc_normal_(self.pos_embed, std=0.02)
    nn.init.trunc_normal_(self.cls_token, std=0.02)
    
  def forward(self, x):
    # размер батча
    B = x.shape[0]
    x = self.patch_embed(x)
    # увеличиваем чтобы для каждого батча был класс токен
    cls = self.cls_token.expand(B, -1, -1)
    # склеили
    x = torch.cat((cls, x), dim=1)
    x = x + self.pos_embed
    # применили внимание на все патчи
    x = self.attn_block(x)
    x = self.norm(x)
    # возвращаем cls и патчи по отдельности
    # инфа по классу улетит в классификатор
    # инфа о патчах пойдёт в декодер
    return x[:, 0:1, :], x[:, 1:, :]
  
class EncoderClassifier(nn.Module):
  def __init__(self, dim, num_classes, dropout) -> None:
    super().__init__()
    self.fc1 = nn.Linear(dim, dim)
    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(dropout)
    self.fc2 = nn.Linear(dim, num_classes)
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, cls_token):
    x = self.fc1(cls_token)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x
  
class CrossAttention(nn.Module):
  def __init__(self, q_dim, ctx_dim, dim, num_heads) -> None:
    super().__init__()
    self.num_heads = num_heads
    # Query (запрос) от декодера
    self.q_proj = nn.Linear(q_dim, dim)
    # Инфа по патчам (key и values), как контекст
    # от енкодера
    self.k_proj = nn.Linear(ctx_dim, dim)
    self.v_proj = nn.Linear(ctx_dim, dim)
    self.out = nn.Linear(dim, q_dim)
    
  def forward(self, q, ctx):
    # Аналог обычного слоя внимания
    B, T, Qd = q.shape
    S = ctx.shape[1]
    d = self.q_proj(q).shape[-1]
    q_ = self.q_proj(q).reshape(B, T, self.num_heads, d // self.num_heads)
    q_ = q_.permute(0, 2, 1, 3)
    k_ = self.k_proj(ctx).reshape(B, S, self.num_heads, d // self.num_heads)
    k_ = k_.permute(0, 2, 1, 3)
    v_ = self.v_proj(ctx).reshape(B, S, self.num_heads, d // self.num_heads)
    v_ = v_.permute(0, 2, 1, 3)
    scale = (d // self.num_heads) ** -0.5
    attn = (q_ @ k_.transpose(-2, -1)) * scale
    attn = attn.softmax(-1)
    out = (attn @ v_).transpose(1, 2).reshape(B, T, d)
    out = self.out(out)
    return out
  
class DecoderBlock(nn.Module):
  def __init__(self, dim, ctx_dim, num_heads) -> None:
    super().__init__()
    self.norm1 = nn.LayerNorm(dim)
    self.self_attn = Attention(dim, num_heads)
    self.norm2 = nn.LayerNorm(dim)
    self.cross_attn = CrossAttention(dim, ctx_dim, dim, num_heads)
    self.norm3 = nn.LayerNorm(dim)
    self.feed_fwd = FeedForward(dim, dim*2)
    
  def forward(self, x, ctx):
    x = self.norm1(x)
    x = x + self.self_attn(x)
    x = self.norm2(x)
    x = x + self.cross_attn(x, ctx)
    x = self.norm3(x)
    x = x + self.feed_fwd(x)
    return x
  
class Decoder(nn.Module):
  def __init__(self, num_patches, dim, ctx_dim, num_heads, patch_size, in_ch) -> None:
    super().__init__()
    # случайные входные данные которые будут обучаться
    self.dec_tokens = nn.Parameter(torch.randn(1, num_patches, dim))
    self.pos = nn.Parameter(torch.zeros(1, num_patches, dim))
    self.attn_block = DecoderBlock(dim, ctx_dim, num_heads)
    self.norm1 = nn.LayerNorm(dim)
    # чтобы вырнеуть в конце патчи, которые можно будет
    # вернуть в фото обратно
    patch_dim = patch_size * patch_size * in_ch
    self.to_patch = nn.Linear(dim, patch_dim)
    
  def forward(self, ctx):
    B = ctx.shape[0]
    x = self.dec_tokens.expand(B, -1, -1) + self.pos
    x = self.attn_block(x, ctx)
    x = self.norm1(x)
    patches = self.to_patch(x)
    return patches
  
class AutoEncoder(nn.Module):
  def __init__(self, img_size, patch_size, in_ch, embed_dims, enc_heads, dec_heads) -> None:
    super().__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.in_ch = in_ch
    self.encoder = Encoder(img_size, patch_size, in_ch, embed_dims, enc_heads)
    num_patches = (img_size // patch_size) ** 2
    self.decoder = Decoder(num_patches, embed_dims, embed_dims, dec_heads,
                           patch_size, in_ch)
    self.classifier = EncoderClassifier(embed_dims, 10, 0.2)
    
  def forward(self, img):
    cls, patches = self.encoder(img)
    
    # не понял почему, вроде классификатора голова как и раньше написана,
    # но добавлиась лишняя размерность типа [256, 1, 10]
    # а должна быть [256, 10], ну просто решейпом пофиксил 
    enc_out = self.classifier(cls.squeeze(1))
    
    context = patches
    pred_patches = self.decoder(context)
    # начинаем елозить размерностями чтобы потом патчи расставить по местам
    B, N, PD = pred_patches.shape
    p = self.patch_size
    x = pred_patches.view(B, N, self.in_ch, p, p)
    gs = self.img_size // p
    x = x.permute(0,2,3,4,1).reshape(B, self.in_ch, p, p, gs, gs)
    x = x.permute(0,1,4,2,5,3).reshape(B, self.in_ch, gs*p, gs*p)
    # тут уже полноценное изображение возвращается
    return enc_out, x
  
def trans_model(device):
  trans_model = AutoEncoder(img_size=28, patch_size=7, in_ch=1,
                          embed_dims=128, enc_heads=2, dec_heads=2).to(device)
  return trans_model