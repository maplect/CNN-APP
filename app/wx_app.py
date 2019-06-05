import sys,os
from app.network import network
from common.deep_convnet import DeepConvNet
from common.functions import *
sys.path.append(os.pardir)
import wx
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
class MINST_APP(wx.Frame):
    def __init__(self,network,parent = None):
        wx.Frame.__init__(self,parent=None, title='MINST_APP', size=(500,400),pos=(100,100),
                          style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        self.network = network
        '''工具栏'''
        self.sbar1 = self.CreateStatusBar()
        self.sbar1.SetStatusText('please draw a num')
        tbar1 = self.CreateToolBar(wx.TB_HORIZONTAL, wx.ID_ANY)
        tbar1.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))
        btn_clear = wx.Button(tbar1, label=u'clear')  # clear button
        tbar1.AddControl(btn_clear)
        btn_preidct = wx.Button(tbar1, label=u'predict')  # predict button
        tbar1.AddControl(btn_preidct)
        self.auto_predict = wx.CheckBox(tbar1, label=u'auto predict')  # auto predict checkbox
        tbar1.AddControl(self.auto_predict)
        tbar1.Realize()  # realize toolbar
        '''布局'''
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        '''画图面板'''
        box_draw = wx.StaticBox(self, label='Draw')
        sizer_draw = wx.StaticBoxSizer(box_draw,wx.VERTICAL)
        draw_panel = wx.Panel(sizer_draw.GetStaticBox(), size=(280,280))
        draw_panel.SetForegroundColour(wx.Colour(0, 0, 0))
        draw_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        draw_panel.SetMaxSize(wx.Size(280, 280))
        sizer_draw.Add(draw_panel)
        sizer.Add(sizer_draw)
        self.draw_panel = draw_panel
        self.draw_panel.Bind(wx.EVT_PAINT,self.OnPaint)
        '''预测面板'''
        box_predict = wx.StaticBox(self,label='Predict Result')
        sizer_predict = wx.StaticBoxSizer(box_predict,wx.VERTICAL)
        self.PredictRateList = []

        '''0'''
        sizer_predict0 = wx.BoxSizer(wx.HORIZONTAL)
        text0 = wx.StaticText(sizer_predict.GetStaticBox(),label=u'0')
        sizer_predict0.Add(text0)
        gauge0 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge0.SetValue(0)
        sizer_predict0.Add(gauge0)
        self.PredictRateList.append(gauge0)
        sizer_predict.Add(sizer_predict0)
        '''1'''
        sizer_predict1 = wx.BoxSizer(wx.HORIZONTAL)
        text1 = wx.StaticText(sizer_predict.GetStaticBox(), label=u'1')
        sizer_predict1.Add(text1)
        gauge1 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge1.SetValue(0)
        sizer_predict1.Add(gauge1)
        self.PredictRateList.append(gauge1)
        sizer_predict.Add(sizer_predict1)

        '''2'''
        sizer_predict2 = wx.BoxSizer(wx.HORIZONTAL)
        text2 = wx.StaticText(sizer_predict.GetStaticBox(), label=u'2')
        sizer_predict2.Add(text2)
        gauge2 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge2.SetValue(0)
        sizer_predict2.Add(gauge2)
        self.PredictRateList.append(gauge2)
        sizer_predict.Add(sizer_predict2)
        '''3'''
        sizer_predict3 = wx.BoxSizer(wx.HORIZONTAL)
        text3 = wx.StaticText(sizer_predict.GetStaticBox(), label=u'3')
        sizer_predict3.Add(text3)
        gauge3 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge3.SetValue(0)
        sizer_predict3.Add(gauge3)
        self.PredictRateList.append(gauge3)
        sizer_predict.Add(sizer_predict3)
        '''4'''
        sizer_predict4 = wx.BoxSizer(wx.HORIZONTAL)
        text4 = wx.StaticText(sizer_predict.GetStaticBox(), label=u'4')
        sizer_predict4.Add(text4)
        gauge4 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge4.SetValue(0)
        sizer_predict4.Add(gauge4)
        self.PredictRateList.append(gauge4)
        sizer_predict.Add(sizer_predict4)
        '''5'''
        sizer_predict5 = wx.BoxSizer(wx.HORIZONTAL)
        text5 = wx.StaticText(sizer_predict.GetStaticBox(), label=u'5')
        sizer_predict5.Add(text5)
        gauge5 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge5.SetValue(0)
        sizer_predict5.Add(gauge5)
        # PredictRateList = []
        self.PredictRateList.append(gauge5)
        sizer_predict.Add(sizer_predict5)
        '''6'''
        sizer_predict6 = wx.BoxSizer(wx.HORIZONTAL)
        text6 = wx.StaticText(sizer_predict.GetStaticBox(), label=u'6')
        sizer_predict6.Add(text6)
        gauge6 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge6.SetValue(0)
        sizer_predict6.Add(gauge6)
        self.PredictRateList.append(gauge6)
        sizer_predict.Add(sizer_predict6)
        '''7'''
        sizer_predict7 = wx.BoxSizer(wx.HORIZONTAL)
        text7 = wx.StaticText(sizer_predict.GetStaticBox(), label=u'7')
        sizer_predict7.Add(text7)
        gauge7 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge7.SetValue(0)
        sizer_predict7.Add(gauge7)
        self.PredictRateList.append(gauge7)
        sizer_predict.Add(sizer_predict7)
        '''8'''
        sizer_predict8 = wx.BoxSizer(wx.HORIZONTAL)
        text8 = wx.StaticText(sizer_predict.GetStaticBox(), label=u'8')
        sizer_predict8.Add(text8)
        gauge8 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge8.SetValue(0)
        sizer_predict8.Add(gauge8)
        self.PredictRateList.append(gauge8)
        sizer_predict.Add(sizer_predict8)
        '''9'''
        sizer_predict9 = wx.BoxSizer(wx.HORIZONTAL)
        text9 = wx.StaticText(sizer_predict.GetStaticBox(), label=u'9')
        sizer_predict9.Add(text9)
        gauge9 = wx.Gauge(sizer_predict.GetStaticBox())
        gauge9.SetValue(0)
        sizer_predict9.Add(gauge9)
        self.PredictRateList.append(gauge9)
        sizer_predict.Add(sizer_predict9)

        '''result'''
        sizer_result = wx.BoxSizer(wx.HORIZONTAL)
        result = wx.StaticText(sizer_predict.GetStaticBox(), label=u'本项目由我爱大蒜鼎力支持',style=wx.ALIGN_CENTER_HORIZONTAL)
        sizer_predict.Add(result)
        sizer.Add(sizer_predict)
        self.SetSizer(sizer)
        self.Layout()
        self.Centre(wx.BOTH)
        self.mnistArray = np.zeros(28*28)
        self.Bind(wx.EVT_CLOSE,self.close)
        btn_clear.Bind(wx.EVT_BUTTON,self.Onclear)
        btn_preidct.Bind(wx.EVT_BUTTON,self.OnPredict)
        self.draw_panel.Bind(wx.EVT_LEFT_DOWN,self.Onleft_down)
        self.draw_panel.Bind(wx.EVT_LEFT_UP, self.Onleft_up)
        self.draw_panel.Bind(wx.EVT_MOTION, self.OnMotion)
        self.isdrawing = False
        self.lastX = 0
        self.lastY = 0
    def OnPaint(self,event):
        dc = wx.PaintDC(self.draw_panel)
        pen = wx.Pen(wx.Colour(100, 100, 100))
        brush = wx.Brush(wx.Colour(255, 155, 155))
        dc.SetPen(pen)
        dc.SetBrush(brush)
        self.drawGrid(dc)
    def drawGrid(self,dc):
        for p in range(0, 280, 10):
            dc.DrawLine(p, 0, p, 280)
            dc.DrawLine(0, p, 280, p)

        for x in range(28):
            for y in range(28):
                d = self.mnistArray[x + y * 28]
                if d != 0:
                    dc.DrawRectangle(x * 10 + 1, y * 10 + 1, 9, 9)
    def Onclear(self,event):
        self.mnistArray = np.zeros(28 * 28)
        for g in self.PredictRateList:
            g.SetValue(0)
        self.sbar1.SetStatusText('Draw a single number above.')
        self.redraw()
    def OnPredict(self,event):
        self.mnistArray = np.reshape(self.mnistArray,(1,1,28,28))
        pd = self.network.predict(self.mnistArray)
        pd = np.reshape(pd,(10,))
        pd = softmax(pd)
        for i in range(len(pd)):
                    self.PredictRateList[i].SetValue(pd[i]*100)
        print(pd)
        self.sbar1.SetStatusText('I guest the number is: %d' % np.argmax(pd))
    def redraw(self):
        self.draw_panel.Refresh()
    def Onleft_down(self,event):
        self.isdrawing = True
    def Onleft_up(self,event):
        self.isdrawing = False
        if self.auto_predict.IsChecked():
            self.OnPredict(event)
    def OnMotion(self,event):
        if self.isdrawing:
            pos = event.Position
            x = int(pos.x / 10)
            y = int(pos.y / 10)
            if x != self.lastX or y != self.lastY:
                self.lastX = x
                self.lastY = y
                self.mnistArray[x + y * 28] = 1
                self.redraw()
    def close(self,event):
        quit()


if __name__ == '__main__':

    network1= DeepConvNet()
    network1.load_params(file_name="deep_convnet_params.pkl")


    app = wx.App()
    APP = MINST_APP(network=network1)
    APP.Show()
    app.MainLoop()
