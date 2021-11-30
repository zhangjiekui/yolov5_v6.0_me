import torch

def test_gather():
    test1 = {"input":torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]]),
            "index": torch.LongTensor([0,2]).view(-1,1),
            'dim':1
            }

    test4 = {"input":torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]]),
            "index": torch.LongTensor([[0,1]]),
            'dim':0,
            }

    input=torch.tensor([[1,2,3],[4,5,6]])
    index1=torch.tensor([[0,1,1],[0,0,1]])

    test2 = {"input":input,
            "index":index1,
            'dim':0
            }

    test3 = {"input":input,
            "index":index1,
            'dim':1
            }


    tests = [test1,test2,test3,test4]

    def torch_gather_test(input,index,dim):
        output_shape=index.shape
        output=torch.ones(output_shape)*-10
        # print(output.shape)
        if dim == 0:
            print("--------------dim0")
            for i in range(output_shape[0]):
                for j in range(output_shape[1]):
                    # print(f"my_{output=}")
                    _i = index[i,j]
                    output[i,j] = input[_i,j]
                    # print(f"{i=},{_i=},{j=},{output[i,j].tolist()=}")
            print(f"my_{output.tolist()= }")
            print("gather           = ",torch.gather(input,dim,index).to(dtype=output.dtype).tolist())


        elif dim == 1:
            print("--------------dim1")
            for i in range(output_shape[0]):
                for j in range(output_shape[1]):
                    # print(f"my_{output=}")
                    _j = index[i,j]
                    output[i,j] = input[i,_j]
                    # print(f"{i=},{_j=},{j=},{output[i,j].tolist()=}")
            print(f"my_{output.tolist()= }")
            # print(f"orignal_{result.tolist()=}")
            print("gather           = ",torch.gather(input,dim,index).to(dtype=output.dtype).tolist())
        else:
            pass

    for test in tests:
        input = test['input']
        index = test['index']
        dim = test['dim']
        torch_gather_test(input,index,dim)

def test_dstack():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    r=torch.dstack((a,b))
    print(a.shape,b.shape,r.shape)
    a = torch.tensor([[1],[2],[3]])
    b = torch.tensor([[4],[5],[6]])
    torch.dstack((a,b))
    print(a.shape,b.shape,r.shape)

def test_dsplit():
    t = torch.arange(16.0).reshape(2, 2, 4)
    t1,t2,_,_=torch.dsplit(t, 4)
    print(t.shape,t1.shape,t2.shape)

def test_index_select():
    a = torch.tensor([[11,12,13],
                [21,22,23],
                [31,32,33]])
    print(a.shape)
    print(f"\n{a=}")
    i=torch.tensor([1,0,2])

    r0=a.index_select(0,i)
    r1=a.index_select(1,i)
    print(f"\n{r0=}")
    print(f"\n{r1=}")

def test_where():
    a=torch.tensor([1,3,5])
    b=torch.tensor([0,4,5])
    w=torch.where(a>b,a,b)
    print(w)
    w=torch.where(a<b,a,b)

    print(w)
def take_along_dim():
    t = torch.tensor([[10, 30, 20], [60, 40, 50]])
    max_idx = torch.argmax(t)
    torch.take_along_dim(t, max_idx)
    sorted_idx = torch.argsort(t, dim=1)
    torch.take_along_dim(t, sorted_idx, dim=1)

def test_repeat_and_tile():
    x = torch.tensor([1, 2, 3])
    a = x.repeat(4, 2)
    b = x.tile(4,2)
    print(f"{a=},\n{b=}")
    print(f"{x.repeat(4, 2, 1).size()=},\n{x.repeat(4, 2, 1)=}\n")
    print(f"{x.tile(4, 2, 1).size()=},\n{x.tile(4, 2, 1)=}\n")

def conv_group_test():
    conv1 = torch.nn.Conv2d(8,4,3)
    conv_g= torch.nn.Conv2d(8,4,3,groups=4)
    print(f'{conv1.weight.shape=}')
    print(f'{conv_g.weight.shape=}')
    print(f'{conv1.bias.shape=}')
    print(f'{conv_g.bias.shape=}')


if __name__ == "__main__":
    # test_gather()
    # test_dstack()

    # test_dsplit()
    # test_index_select()

    # test_where()
    # take_along_dim()
    # test_repeat_and_tile()
    conv_group_test()