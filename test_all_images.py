import os
import xlwt
from xlwt import Workbook
import primer as prm
  
# Workbook is created
wb = Workbook()
  
# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1')
  


#path = 'C:/Users/dband/Documents/handwritten_character_examples/'
path = "handwritten_character_examples/"
files = os.listdir(path)

name_col = 0
quality_col = 1
cnn_char_col = 2
p_col = 3
s_col = 4
g_col = 5

sheet1.write(0, 0, 'Character')
sheet1.write(0, 1, 'Quality')
sheet1.write(0, 2, 'CNN')
sheet1.write(0, 3, 'P')
sheet1.write(0, 4, 'S')
sheet1.write(0, 5, 'G')

counter = 1

for f in sorted(files, key=str.lower):
	#print(f)
    

    filename, ending = f.split(".")
    print(filename)
    print(ending)
    if (ending == "png" or ending == "jpeg" or ending == "jpg"):
        name, quality = filename.split("_")
        p, s, g, cnn_char = prm.run(path+f)


        sheet1.write(counter, name_col, name)
        sheet1.write(counter, quality_col, quality)
        sheet1.write(counter, cnn_char_col, cnn_char)
        sheet1.write(counter, p_col, float(p))
        sheet1.write(counter, s_col, s)
        sheet1.write(counter, g_col, g)

        counter +=1
        #if counter == 50:
        #    break
        wb.save('Handwriting Output.xls')


  
wb.save('Handwriting Output.xls')