document.querySelector('button').addEventListener('click', (e) => {
  e.preventDefault();
  
  params = {};

  const inputs = document.querySelectorAll('input');
  for (const input of inputs) {
    params[input.id] = input.value;
  }

  const textareas = document.querySelectorAll('textarea');
  for (const textarea of textareas) {
    params[textarea.id] = textarea.value;
  }

  // save to local storage
  localStorage.setItem('params', JSON.stringify(params));

  // const url = `?waveform=${waveform}&frequency=${frequency}&amplitude=${amplitude}&phase=${phase}&offset=${offset}&samplerate=${samplerate}&samples=${samples}`;
  // window.open(url);
});

// set default values from local storage on page load
window.addEventListener('load', () => {
  const params = JSON.parse(localStorage.getItem('params'));
  if (params) {
    for (const key in params) {
      const element = document.getElementById(key);
      if (element) {
        element.value = params[key];
      }
    }
  }
});