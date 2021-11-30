 
print("MMMMMMMMMMMMMMMMMM")
import torch
x = torch.randn(2, 3)
y = torch.randn(3, 4)

m = torch.einsum("ik, kj->ij", x, y)
n = torch.einsum("ik, kj->ji", x, y)
print("a:{} \nb:{}".format(m, n))

# Output：
# a: tensor([[-1.0836, -0.2650, -1.7384, -0.5368],
#            [1.1246, -0.2049, 1.5340, 0.6870]])
# b: tensor([[-1.0836, 1.1246],
#            [-0.2650, -0.2049],
#            [-1.7384, 1.5340],
#            [-0.5368, 0.6870]])

print("MMMMMMMMMMMMMMMMMMNNNNNNNNNNNNNNN")
print(m.shape,n.shape)
r_i=torch.einsum('ij,jk->i',m,n)
print(r_i.shape)

r_j=torch.einsum('ij,jk->j',m,n)
print(r_j.shape)

r_ik=torch.einsum('ij,jk->ik',m,n)
print(r_ik.shape)

r_ik=torch.einsum('ij,jk',m,n)
print(r_ik.shape)

r=torch.einsum('ij,jk->',m,n)
print(r.shape,r)
print("=======================================")
a3=torch.randn(3)
b5=torch.rand(5)
print(f"{a3=},{a3.shape=}")
print(f"{b5=},{b5.shape=}")
r=torch.einsum('i,j->ij',a3,b5)
print(f"{r=},{r.shape=}")
print("diagnal=======================================")
diagnal = torch.randn((3,3))
print(f"{diagnal=},{diagnal.shape=}")
d=torch.einsum('ii -> i',diagnal)
print(f"{d=},{d.shape=}")

dt=torch.einsum('ii ->',diagnal)
print(f"{dt=},{dt.shape=},{d.sum()}")