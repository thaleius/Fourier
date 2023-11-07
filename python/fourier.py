from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def deg2rad(deg):
  return deg * np.pi / 180

class Param:
  def __init__(self, name, value):
    self.name = name
    self.value = value

  def __str__(self):
    return str(self.value)
  
  def __repr__(self):
    return repr(self.value)
  
  def __add__(self, other):
    return self.value + other
  
  def __radd__(self, other):
    return self.value + other
  
  def __sub__(self, other):
    return self.value - other
  
  def __rsub__(self, other):
    return self.value - other
  
  def __mul__(self, other):
    return self.value * other
  
  def __rmul__(self, other):
    return self.value * other
  
  def __truediv__(self, other):
    return self.value / other

class ParamList:
  params = Param

  def __init__(self, *params):
    self.params = params

  def __getitem__(self, key):
    if type(key) is not int and type(key) is not str:
      raise Exception('Invalid type. Must be an integer or a string')
    if type(key) is str:
      for i in range(len(self.params)):
        if self.params[i].name == key:
          return self.params[i]
      raise Exception('Invalid key. Must be a valid parameter name')
    elif type(key) is int:
      if key < 0 or key >= len(self.params):
        raise Exception('Invalid index. Must be between 0 and ' + str(len(self.params) - 1))
      return self.params[key]
  
  def __len__(self):
    return len(self.params)
  
  def __iter__(self):
    return iter(self.params)
  
  def __contains__(self, item):
    return item in self.params
  
  def __str__(self):
    return str(self.params)
  
  def __repr__(self):
    return repr(self.params)

class Signal:
  aliases = {
    'sine': 'sine',
    'cosine': 'cosine',
    'square': 'square',
    'sawtooth': 'sawtooth',

    'sin': 'sine',
    'cos': 'cosine',
    'squ': 'square',
    'saw': 'sawtooth',

    'sinus': 'sine',
    'cosinus': 'cosine',
    'rechteck': 'square',
    'sägezahn': 'sawtooth',
  }

  def __init__(self, function = None, frequency = 1, amplitude = 1, phase = 0, offset = 0, **kwargs):
    self.params = ParamList(Param('frequency', frequency), Param('amplitude', amplitude), Param('phase', phase), Param('offset', offset))
    
    for param in self.params:
      if param.name in kwargs:
        param.value = kwargs[param.name]

    if callable(function):
      self.f = function
      self.waveform = 'custom'
    elif type(function) is str:
      if function.lower() not in self.aliases:
        raise Exception('Invalid type. Valid types are: ' + ', '.join(self.aliases.keys()))
      
      self.waveform = function.lower()

      if self.waveform == 'sine':
        self.f = lambda t: self.sine(self.params['amplitude'], self.params['frequency'], self.params['phase'], self.params['offset'], t)
      elif self.waveform == 'cosine':
        self.f = lambda t: self.cosine(self.params['amplitude'], self.params['frequency'], self.params['phase'], self.params['offset'], t)
      elif self.waveform == 'square':
        self.f = lambda t: self.square(self.params['amplitude'], self.params['frequency'], self.params['phase'], self.params['offset'], t)
        # self.f = lambda t: self.square(amplitude, frequency, phase, offset, t+1/samplerate), samplerate, samples)
      elif self.waveform == 'sawtooth':
        self.f = lambda t: self.sawtooth(self.params['amplitude'], self.params['frequency'], self.params['phase'], self.params['offset'], t-1/(2*self.params['frequency']))
    else:
      raise Exception('Invalid type. Must be a function or a string')
    if 'amplitude' in kwargs:
      self.amplitude = kwargs['amplitude']
    if 'frequency' in kwargs:
      self.frequency = kwargs['frequency']
    if 'phase' in kwargs:
      self.phase = kwargs['phase']
    if 'offset' in kwargs:
      self.offset = kwargs['offset']

  def sine(self, amplitude, frequency, phase, offset, t):
    phi = deg2rad(phase)
    return amplitude * np.sin(2 * np.pi * frequency * t + phi) + offset

  def cosine(self, amplitude, frequency, phase, offset, t):
    phi = deg2rad(phase)
    return amplitude * np.cos(2 * np.pi * frequency * t + phi) + offset

  def square(self, amplitude, frequency, phase, offset, t):
    phi = deg2rad(phase)
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phi)) + offset

  def triangle(self, amplitude, f, phase, offset, t):
    phi = phase / 180
    return amplitude * np.abs((4 * f * t + phi) % 4 - 2) - 1 + offset

  def sawtooth(self, amplitude, frequency, phase, offset, t):
    phi = phase / 180
    return amplitude * ((2 * frequency * t + phi) % 2 - 1) + offset

  def __call__(self, t):
    return self.f(t)
  
  def __add__(self, other):
    if type(other) is not Signal:
      raise Exception('Invalid type. Must be a function')

    return Signal(lambda t: self.f(t) + other.f(t))
  
  def __sub__(self, other):
    if type(other) is not Signal:
      raise Exception('Invalid type. Must be a function')

    return Signal(lambda t: self.f(t) - other.f(t))
  
  def sample(self, samplerate = 1000, samples = -1, **kwargs):
    if type(samplerate) is not int and type(samplerate) is not float:
      raise Exception('Invalid samplerate type. Must be a number')
    if type(samples) is not int and type(samples) is not float:
      raise Exception('Invalid samples type. Must be a number')
    
    if samples == -1:
      samples = samplerate

    if 'samplerate' in kwargs:
      samplerate = kwargs['samplerate']
    if 'samples' in kwargs:
      samples = kwargs['samples']

    self.samplerate = samplerate
    self.samples = samples
    data = [(t, self(t + (1/samplerate if self.waveform == 'square' else 0))) for t in np.linspace(0, ((samples-1)/samplerate), samples)]
    return Data(pd.DataFrame(data, columns=['t', 'Signal']), samplerate=samplerate, samples=samples)

class DataB:
  def __init__(self, data, **kwargs):
    if type(data) is not pd.DataFrame and type(data) is not str:
      raise Exception('Invalid type. Must be a pandas DataFrame or a path to a csv file')
    
    if type(data) is pd.DataFrame:
      if data.empty:
        raise Exception('Invalid data. Must not be empty')
      if len(data.columns) < 2:
        raise Exception('Invalid data. Must have at least 2 columns')

      self.data = data
    elif type(data) is str:
      if '\n' in data:
        self.data = pd.DataFrame([row.replace(',', '.').replace(kwargs['decimal'] if 'decimal' in kwargs else '.', '.').split(kwargs['sep'] if 'sep' in kwargs else '\t') for row in data.split('\n')[1:]], columns=data.split('\n')[0].split(kwargs['sep'] if 'sep' in kwargs else '\t'), dtype=np.float64)
      else:
        self.data = pd.read_csv(data, sep=kwargs['sep'] if 'sep' in kwargs else '\t', header=1, names=kwargs['names'] if 'names' in kwargs else ['t', 'Signal'], decimal=kwargs['decimal'] if 'decimal' in kwargs else '.', dtype={'t': np.float64, 'Signal': np.float64})

    self.params = kwargs
    
  def print(self):
    if self.data is None:
      raise Exception('No data to print')

    print(self.data)
    return self

  def save(self, path: str = 'data.csv'):
    if self.data is None or not isinstance(self.data, pd.DataFrame) or self.data.empty:
      raise Exception('No data to save')

    self.data.to_csv(path, index=False)

    return self

  def to_string(self, *args, **kwargs):
    return self.data.to_string(*args, **kwargs)
  
  def __str__(self):
    return str(self.data)
  
  def __repr__(self):
    return repr(self.data)
  
  def __getitem__(self, key):
    return self.data[key]
  
  def to_string(self, *args, **kwargs):
    return self.data.to_string(*args, **kwargs)

  def to_html(self, *args, **kwargs):
    table = '<table>'
    table += '<tr><th>' + '</th><th>'.join(self.data.columns) + '</th></tr>'
    for row in self.data.values:
      table += '<tr><td>' + '</td><td>'.join([str(col).replace('.', ',') for col in row]) + '</td></tr>'
    table += '</table>'
    return table
  
class DataF(DataB):
  def __init__(self, data, **kwargs):
    self.data = data

    if 'samplerate' in kwargs:
      self.samplerate = kwargs['samplerate']
    if 'samples' in kwargs:
      self.samples = kwargs['samples']

    self.params = kwargs

  def plot(self, **kwargs):
    if self.data is None:
      raise Exception('No data to plot')

    fig, ax = plot()
    if 'real' not in kwargs or kwargs['real'] != 0:
      ax.vlines(self.data['f'], 0, self.data['A'].apply(lambda x: np.real(x)), label='Kosinus', colors='green')
    if 'imag' not in kwargs or kwargs['imag'] != 0:
      ax.vlines(self.data['f'], 0, self.data['A'].apply(lambda x: np.imag(x)), label='Sinus', colors='red')

    if kwargs['legend'] if 'legend' in kwargs else True:
      # put legend at best position
      ax.legend(loc='best')

    ax.set_xlabel(r'Frequenz / Hz')
    ax.set_ylabel(r'Amplitude')

    ax.set_xlim(*kwargs['xlim'] if 'xlim' in kwargs else (None, None))
    ax.set_ylim(*kwargs['ylim'] if 'ylim' in kwargs else (None, None))

    return fig
  
  # inverse fourier transform
  def ifft(self, **kwargs):    
    if 'samplerate' in kwargs:
      sampling_rate = kwargs['samplerate']
    else:
      sampling_rate = 1000
    if 'samples' in kwargs:
      samples = kwargs['samples']
    else:
      samples = 1000

    real = True
    imag = True

    if 'real' in kwargs:
      if kwargs['real'] == 0:
        real = False        
    if 'imag' in kwargs:
      if kwargs['imag'] == 0:
        imag = False

    if real:
      signal = Signal('cosine', frequency=self.data.iloc[0]['f'], amplitude=np.real(self.data.iloc[0]['A']))
    if imag:
      signal = Signal('sine', frequency=self.data.iloc[0]['f'], amplitude=-np.imag(self.data.iloc[0]['A']))
    for i in range(1, len(self.data)):
      if real:
        signal += Signal('cosine', frequency=self.data.iloc[i]['f'], amplitude=np.real(self.data.iloc[i]['A']))
      if imag:
        signal += Signal('sine', frequency=self.data.iloc[i]['f'], amplitude=-np.imag(self.data.iloc[i]['A']))

    return signal.sample(sampling_rate, samples)

class Data(DataB):
  def plot(self, draw_dots = True, *args, **kwargs):
    if self.data is None:
      raise Exception('No data to plot')

    self.data[self.data.columns[1]] = self.data[self.data.columns[1]].apply(lambda x: np.real(x))

    fig, ax = plot()
    ax.plot(self.data[self.data.columns[0]], self.data[self.data.columns[1]], label='Signal')
  
    if draw_dots:
      ax.plot(self.data[self.data.columns[0]], self.data[self.data.columns[1]], '.', label='Abtastwerte')
    
    if kwargs['legend'] if 'legend' in kwargs else True:
      ax.legend()
    ax.set_xlabel(r'Zeit / s')
    ax.set_ylabel(r'Amplitude')
    
    ax.set_xlim(*kwargs['xlim'] if 'xlim' in kwargs else (None, None))
    ax.set_ylim(*kwargs['ylim'] if 'ylim' in kwargs else (None, None))

    return fig

  def fft(self):
    from scipy.fft import fft, fftfreq, rfft, rfftfreq

    fourier = fft(self.data[self.data.columns[1]].values)

    N = len(self.data[self.data.columns[1]].values)
    normalize = N/2

    sampling_rate = self.params['samplerate']

    df = pd.DataFrame({'f': rfftfreq(N, d=1/sampling_rate), 'A': 2*(rfft(self.data[self.data.columns[1]].values))/N})#[1:]

    return DataF(df, samplerate=sampling_rate, samples=N)