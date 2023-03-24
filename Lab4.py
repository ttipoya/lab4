#26.	Формируется матрица F следующим образом: скопировать в нее А и  если в С количество нулей в нечетных столбцах больше,
# чем произведение чисел по периметру , то поменять местами  В и С симметрично, иначе В и Е поменять местами несимметрично.
# При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F,
# то вычисляется выражение: A-1*AT – K * F, иначе вычисляется выражение (A +G-1-F-1)*K, где G-нижняя треугольная матрица, полученная из А.
# Выводятся по мере формирования А, F и все матричные операции последовательно.
import numpy as np
import matplotlib.pyplot as plt
print("Введите число N не меньше 6:", end='')
n = int(input())
print("Введите число K:", end='')
k = int(input())
K = [k]
A = np.random.randint(-10,10, size=(n,n),dtype="int64")
Amo = np.copy(A)
F = np.copy(A)
At = np.zeros((n,n))
Ft = np.zeros((n,n))
col1 = 0
col2 = 1
prav = 0
pr1 = 1
pr2 = 1
su = 0
SUM = 0
if n >= 6:
    print("\nОсновная матрица:")
    print(A)
    for i in range(n//2, n):
        if F[0,i] == 0:
            col1 += 1
        col2 *= F[0,i]
    for i in range(1, n//2):
        if F[i,n//2] == 0:
            col1 += 1
        col2 *= F[i,n//2]
    for i in range((n//2)+1, n):
        if F[(n//2)-1,i] == 0:
            col1 += 1
        col2 *= F[(n//2)-1,i]
    for i in range(1, (n//2)-1):
        if F[i,n-1] == 0:
            col1 += 1
        col2 *= F[i,n-1]
    print("\nКол-во нулей:",col1)
    print("Произведение:",col2,'\n')
    if col1 > col2:
        for i in range(n//2):
            for j in range(n//2,n-1):
                F[i,j],F[(n-1)-i,j] = F[(n-1)-i,j],F[i,j]
    elif col1 < col2:
        for i in range(0,(n//2)-1):
            for j in range(n//2):
                F[i, j], F[i, j+ (n//2)] = F[i, j+ (n//2)], F[i, j]
    elif col1 == col2:
        print("Условие не выполнено")
        prav +=1
    if prav == 0:
        print('Матрица F')
        print(F)
        SUM = int(np.linalg.det(A))
        for i in range(0,n):
            su = su + F[i,i]
        print(' ')
        for i in range(0,n):
            su = su + F[i, n - 1 - i]
        print('Определитель A:', SUM)
        print('Сумма диагональных чисел F:', su)
        print('')
        SUM = [1 / SUM]
        if SUM > su:
            print('A-1')
            Amo = np.multiply(SUM, A)
            print(Amo)
            print(' ')
            print("At")
            At = np.transpose(A)
            print(At)
            print(' ')
            print("A-1 * At")
            Proiz = np.dot(Amo,At)
            print(Proiz)
            print(' ')
            print("(A-1 * At) - (K * F)")
            Raz = (Proiz - (np.multiply(K,F)))
            print(Raz)
        elif SUM < su:
            print('G')
            G = np.copy(np.tril(A))
            print(G)
            print(' ')
            print('F-1')
            Fmo = (int(np.linalg.det(A)) * F)
            print(Fmo)
            print('(A + G - F-1)* K')
            Proz = np.multiply((A + G + Fmo),K)
            print(Proz)
        grap1 = [np.mean(abs(F[i, ::])) for i in range(n)]
        grap1 = int(sum(grap1))
        fixg, grap2 = plt.subplots(2, 2, figsize=(11, 8))
        x = list(range(1, n+1))
        for j in range(n):
            y = list(F[j, ::])
            grap2[0, 0].plot(x, y, ',-', label=f"{j + 1} строка.")
            grap2[0, 0].set(title="График с использованием функции plot:", xlabel='Номер элемента в строке', ylabel='Значение элемента')
            grap2[0, 0].grid()
            grap2[0, 1].bar(x, y, 0.4, label=f"{j + 1} строка.")
            grap2[0, 1].set(title="График с использованием функции bar:", xlabel='Номер элемента в строке', ylabel='Значение элемента')
            if n <= 10:
                grap2[0, 1].legend(loc='lower right')
                grap2[0, 1].legend(loc='lower right')
        exp = [0] * (n - 1)
        exp.append(0.1)
        sizes = [round(np.mean(abs(F[i, ::])) * 100 / grap1, 1) for i in range(n)]
        grap2[1, 0].set_title("График с ипользованием функции pie:")
        grap2[1, 0].pie(sizes, labels=list(range(1, n + 1)), explode=exp, autopct='%1.1f%%', shadow=True)
        def map(data, row_labels, col_labels, grap3, bar_gh={}, **kwargs):
            da = grap3.imshow(data, **kwargs)
            bar = grap3.figure.colorbar(da, ax=grap3, **bar_gh)
            grap3.set_xticks(np.arange(data.shape[1]), labels=col_labels)
            grap3.set_yticks(np.arange(data.shape[0]), labels=row_labels)
            return da, bar
        def annoheat(da, data = None, textcolors=("black", "white"), threshold=0):
            if not isinstance(data, (list, np.ndarray)):
                data = da.get_array()
            gh = dict(horizontalalignment="center", verticalalignment="center")
            texts = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    gh.update(color=textcolors[int(data[i, j] > threshold)])
                    text = da.axes.text(j, i, data[i, j], **gh)
                    texts.append(text)
            return texts
        da, bar = map(F, list(range(n)), list(range(n)), grap3=grap2[1, 1], cmap="magma_r")
        texts = annoheat(da)
        grap2[1, 1].set(title="Создание аннотированных тепловых карт:", xlabel="Номер столбца", ylabel="Номер строки")
        plt.suptitle("Использование библиотеки matplotlib")
        plt.tight_layout()
        plt.show()
else:
    print('Введённое число некорректно')

