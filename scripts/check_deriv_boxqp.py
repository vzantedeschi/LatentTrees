import numpy as np
import torch


def closed_form_boxqp_val(q, b):
    # max <z, q> - .5 |z|^2
    # st 0 <= z <= b

    out = q.clone()
    out[out <= 0] = 0
    out[out >= b] = b[out >= b]
    return 0.5 * torch.dot(out, out) - torch.dot(out, q)


def closed_form_boxqp(q, b):
    # max <z, q> - .5 |z|^2
    # st 0 <= z <= b

    out = q.clone()
    out[out <= 0] = 0
    out[out >= b] = b[out >= b]
    return out


def main():
    n = 5
    q = torch.randn(n)
    print(q)
    b = torch.tensor([.1, .2, 1, 1, 1], requires_grad=True)

    val = closed_form_boxqp_val(q, b)


    print(torch.autograd.grad(val, b, retain_graph=True))


    z = closed_form_boxqp(q, b)

    print(z)

    for i in range(n):
        print(torch.autograd.grad(z[i], b, retain_graph=True))



if __name__ == '__main__':
    main()

