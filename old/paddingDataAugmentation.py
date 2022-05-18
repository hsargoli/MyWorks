# ============
# PADDING
from PIL import Image
test0=Image.open('d://Path//lena.jpg')
width,height=test0.size
tsh=512
left=(-width+tsh)/2
right=(-width+tsh)/2
top=(-height+tsh)/2
bot=(-height+tsh)/2
new_width=int(width+left+right)
new_height=int(height+top+bot)
result=Image.new(test0.mode,(new_width,new_height),(255,255,255))
result.paste(test0,(int(left),int(top)))
result
