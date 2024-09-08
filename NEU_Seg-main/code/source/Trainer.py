import torch
import torch.utils
import torch.utils.data
import torch.nn as nn

import streamlit as st
import plotly.graph_objs as go
from d2l import torch as d2l

d2l.train_ch13


def try_all_gpus():
    devices = [
        torch.device(f'cuda:{i}')
        for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            if isinstance(x, list):
                x = [_x.to(device) for _x in x]
            else:
                x = x.to(device)
            y.squeeze(1)
            y = y.to(device)
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


def train_batch(net, x, y, loss, trainer, devices):
    if isinstance(x, list):
        x = [_x.to(devices[0]) for _x in x]
    else:
        x = x.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(x)
    y = y.squeeze(1).long()
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train(net, train_iter, test_iter, loss, trainer, num_epochs, devices=try_all_gpus()):
    num_batches = len(train_iter)
    animator = Animator('Saikr', 'Train evaluate', [1, num_epochs], x_lable='epoch')
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            if (i + 1) % (num_batches // 4) == 0 or i == num_batches - 1:
                animator.add('loss', epoch + (i + 1) / num_batches, metric[0] / metric[2])
                animator.add('accuracy', epoch + (i + 1) / num_batches, metric[1] / metric[3])

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add('test_accuracy', epoch + 1, test_acc)

    sentence = f'loss {metric[0] / metric[2]:.3f}, train accuracy {metric[1] / metric[3]:.3f}, test accuracy {test_acc:.3f} \n On {str(devices)}'
    animator.text(sentence)


class Animator:
    def __init__(self, web_title, fig_title, range, x_lable='', y_lable=''):
        st.title(web_title)
        self.color = ['brown', 'darkblue', 'cyan']
        self.colorCount = 0
        self.trace = {}
        self.fig = go.Figure()
        self.fig.update_layout(
            xaxis=dict(range=range),
            title=fig_title,
            xaxis_title=x_lable,
            yaxis_title=y_lable
        )
        self.stream = st.plotly_chart(self.fig)

    def add(self, trace_name, x, y):
        if trace_name in self.trace:
            x_ = self.trace[trace_name]['x']
            x_.append(x)
            y_ = self.trace[trace_name]['y']
            y_.append(y)
            showlegend = False
        else:
            self.trace[trace_name] = {'x': [x], 'y': [y], 'color': self.color[self.colorCount]}
            self.colorCount += 1
            x_ = self.trace[trace_name]['x']
            y_ = self.trace[trace_name]['y']
            showlegend = True

        color = self.trace[trace_name]['color']
        trace_new = go.Scatter(
            x=tuple(x_),
            y=tuple(y_),
            mode='lines+markers',
            text=trace_name,
            textposition="top center",
            name=trace_name,
            showlegend=showlegend,
            line=dict(color=color)
        )
        self.fig.add_trace(trace_new)
        self.stream.plotly_chart(self.fig)

    def text(self, sentence):
        st.text(sentence)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
