from sklearn import metrics

y_true = ['fake', 'fake', 'legitimate']
y_pred = ['fake', 'legitimate', 'fake']

# 计算类别为'fake'的准确率
precision_fake = metrics.precision_score(y_true, y_pred, pos_label='fake')

# 计算类别为'legitimate'的准确率
precision_legitimate = metrics.precision_score(y_true, y_pred, pos_label='legitimate')

print("类别'fake'的准确率：", precision_fake)
print("类别'legitimate'的准确率：", precision_legitimate)
