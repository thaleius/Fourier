document.querySelector('button').addEventListener('click', (e) => {
  e.preventDefault();
  
  params_list = [];

  const groups = document.querySelectorAll('#groups .group');
  for (const group of groups) {
    const params = {};
    const inputs = group.querySelectorAll('input');
    for (const input of inputs) {
      params[input.className] = input.value;
    }

    const textareas = group.querySelectorAll('textarea');
    for (const textarea of textareas) {
      params[textarea.className] = textarea.value;
    }

    params_list.push(params);
  }

  // save to local storage
  localStorage.setItem('params', JSON.stringify(params_list));
});

document.querySelector('button#add').addEventListener('click', (e) => {
  e.preventDefault();

  const groups = document.querySelector('#groups');
  const group = groups.querySelector('.group').cloneNode(true);

  const remove = document.createElement('button');
  remove.className = 'remove';
  remove.innerHTML = 'Remove';
  remove.addEventListener('click', (e) => {
    e.preventDefault();
    group.remove();
  });
  group.appendChild(remove);

  groups.appendChild(group);
});

// set default values from local storage on page load
window.addEventListener('load', () => {
  const params = JSON.parse(localStorage.getItem('params'));
  console.log(params);
  if (params) {
    const groups = document.querySelectorAll('#groups .group');
    for (let i = 0; i < groups.length; i++) {
      const inputs = groups[i].querySelectorAll('input');
      for (const input of inputs) {
        input.value = params[i][input.className];
      }

      const textareas = groups[i].querySelectorAll('textarea');
      for (const textarea of textareas) {
        textarea.value = params[textarea.className];
      }
    }
  }
});