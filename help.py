from scipy import sparse
#define text file to open
filename = 'n10k.txt'



with open(filename) as file:
    i = 0
    inp_str = str(file.readline())
    Nrows = [int(x) for x in inp_str.split() if x.isdigit()]
    Nrows = Nrows[0]
    print(Nrows)
    inp_str = str(file.readline())
    nnz = [int(x) for x in inp_str.split() if x.isdigit()]
    nnz = nnz[0]
    

print("nnz is ",nnz," nrows is ",Nrows)

