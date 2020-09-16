'''
code1
'''
print(1)
print(2)
print(3)


'''
code2
'''
a = ['A','B','C']
for i in a:
    print(i)

    
'''
code3
'''
for i in range(1,6,1):
    print(i)
    
    
'''
code4
'''
for i in range(5,0,-1):
    print(i)
    
    
'''
code5
'''
count = 0
for i in range(1,6,1):
    count += 1
    print(count)
    

'''
code6
'''
count = 0
for i in range(1,6,1):
    count += 1
print(count/5)


'''
code7
'''
tmrlist=['今日頭條新聞，TMR公司開設行銷資料科學專班', '好棒棒', 'TMR課程好']
for i in tmrlist:
    if 'TMR' in i:
        print(i)