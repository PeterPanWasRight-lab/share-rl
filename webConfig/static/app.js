'use strict';

var currentConfigName = null;
var currentSummary = null;
var currentRaw = null;
var transitionTypeMeta = {};
var primitiveTypeMeta = {};
var cy = null;

function byId(id) { return document.getElementById(id); }
function deepCopy(value) { return JSON.parse(JSON.stringify(value)); }
function option(value, label) {
  var element = document.createElement('option');
  element.value = value;
  element.textContent = label || value;
  return element;
}

function showToast(message, type) {
  var toast = byId('toast');
  toast.textContent = message;
  toast.className = 'toast ' + (type || 'info') + ' show';
  window.setTimeout(function () { toast.classList.remove('show'); }, 3500);
}

async function api(url, options) {
  var response;
  try {
    response = await fetch(url, Object.assign({
      headers: { 'Content-Type': 'application/json' }
    }, options || {}));
    var text = await response.text();
    var payload = text ? JSON.parse(text) : {};
    if (!response.ok) throw new Error(payload.error || payload.message || ('HTTP ' + response.status));
    return payload;
  } catch (error) {
    showToast(error.message || String(error), 'error');
    throw error;
  }
}

function setLoaded(data) {
  currentConfigName = data.name;
  currentSummary = data.summary;
  currentRaw = data.raw;
  updateGlobalSettings();
  renderGraph();
  showDetailPlaceholder();
}

async function loadConfigList() {
  var names = await api('/api/configs');
  var list = byId('config-list');
  list.replaceChildren();
  names.forEach(function (name) {
    var item = document.createElement('div');
    item.className = 'config-item' + (name === currentConfigName ? ' active' : '');
    var label = document.createElement('span');
    label.textContent = name;
    label.onclick = function () { loadConfig(name); };
    var remove = document.createElement('button');
    remove.className = 'delete-btn';
    remove.type = 'button';
    remove.title = '删除 ' + name;
    remove.textContent = '✕';
    remove.onclick = async function (event) {
      event.stopPropagation();
      if (!window.confirm('确认删除配置“' + name + '”？此操作无法撤销。')) return;
      await api('/api/configs/' + encodeURIComponent(name), { method: 'DELETE' });
      if (name === currentConfigName) {
        currentConfigName = null;
        currentSummary = null;
        currentRaw = null;
        clearGraph();
        showDetailPlaceholder();
      }
      await loadConfigList();
      showToast('已删除 ' + name, 'success');
    };
    item.append(label, remove);
    list.appendChild(item);
  });
  return names;
}

async function loadConfig(name) {
  var data = await api('/api/configs/' + encodeURIComponent(name));
  setLoaded(data);
  await loadConfigList();
  showToast('已加载：' + name, 'info');
}

function syncGlobalSettings() {
  if (!currentRaw) return;
  currentRaw.start_primitive = byId('sel-start-primitive').value;
  currentRaw.reset_primitive = byId('sel-reset-primitive').value;
  currentRaw.fps = Number(byId('inp-fps').value);
}

function updateGlobalSettings() {
  if (!currentSummary) return;
  var names = currentSummary.primitives.map(function (primitive) { return primitive.name; });
  ['sel-start-primitive', 'sel-reset-primitive'].forEach(function (id) {
    var select = byId(id);
    select.replaceChildren();
    names.forEach(function (name) { select.appendChild(option(name)); });
  });
  byId('sel-start-primitive').value = currentSummary.start_primitive;
  byId('sel-reset-primitive').value = currentSummary.reset_primitive;
  byId('inp-fps').value = currentSummary.fps;
}

function clearGraph() {
  if (cy) cy.destroy();
  cy = null;
}

function edgeLabel(transition) {
  var parameters = transition.parameters || {};
  if (transition.type === 'always') return '→';
  if (transition.type === 'on_time_limit') return '⏱ ' + parameters.max_steps + ' 步';
  if (transition.type === 'on_target_pose_reached') return '🎯 ' + ((parameters.axes || ['all']).join(','));
  if (transition.type === 'on_success') return '✓ ' + (parameters.success_key || 'success');
  if (transition.type === 'on_event') return '⚡ ' + (parameters.event_key || 'event');
  if (transition.type === 'on_observation_threshold') return '阈值 ' + (parameters.operator || 'ge') + ' ' + parameters.threshold;
  if (transition.type === 'reward_classifier') return '分类器 ≥ ' + parameters.threshold;
  return transition.type;
}

function renderGraph() {
  clearGraph();
  if (!currentSummary) return;
  if (typeof window.cytoscape !== 'function') {
    showToast('Cytoscape.js 加载失败，请检查网络或 CDN', 'error');
    return;
  }
  var elements = [];
  currentSummary.primitives.forEach(function (primitive) {
    var roles = primitive.roles || {};
    var adaptive = Object.values(primitive.task_frames || {}).some(function (frame) {
      return Number(frame.policy_action_dim || 0) > 0;
    });
    var classes = [adaptive ? 'rl' : 'scripted'];
    if (roles.is_start) classes.push('start');
    if (roles.is_reset) classes.push('reset');
    if (roles.is_terminal) classes.push('terminal');
    elements.push({
      group: 'nodes',
      data: { id: primitive.name, label: primitive.name, primitive: primitive, adaptive: adaptive },
      classes: classes.join(' ')
    });
  });
  currentSummary.transitions.forEach(function (transition) {
    elements.push({
      group: 'edges',
      data: {
        id: 'transition-' + transition.index,
        source: transition.source,
        target: transition.target,
        label: edgeLabel(transition),
        transition: transition
      }
    });
  });
  var layout = { name: typeof window.dagre === 'object' ? 'dagre' : 'breadthfirst', directed: true, rankDir: 'LR', padding: 30 };
  cy = window.cytoscape({
    container: byId('cy'), elements: elements, layout: layout,
    minZoom: 0.25, maxZoom: 3, wheelSensitivity: 0.25,
    style: [
      { selector: 'node', style: { label: 'data(label)', 'text-valign': 'center', color: '#fff', 'background-color': '#4a9eff', width: 'label', height: 36, padding: 14, shape: 'roundrectangle', 'border-width': 2, 'border-color': '#4a9eff', 'font-size': 11 } },
      { selector: 'node.rl', style: { 'background-color': '#ffc940', 'border-color': '#ffc940', color: '#1a1b2e' } },
      { selector: 'node.start', style: { 'border-color': '#4caf50', 'border-width': 5 } },
      { selector: 'node.terminal', style: { 'border-color': '#ff5252', 'border-width': 5, 'border-style': 'double' } },
      { selector: 'edge', style: { label: 'data(label)', width: 2, 'line-color': '#606080', 'target-arrow-color': '#606080', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'font-size': 10, color: '#a0a0c0', 'text-background-color': '#1a1b2e', 'text-background-opacity': 0.9, 'text-background-padding': 3, 'text-rotation': 'autorotate' } },
      { selector: ':selected', style: { 'border-color': '#fff', 'line-color': '#fff', 'target-arrow-color': '#fff' } }
    ]
  });
  cy.on('tap', 'node', function (event) { showNodeDetail(event.target.data()); });
  cy.on('tap', 'edge', function (event) { showEdgeDetail(event.target.data().transition); });
  cy.on('tap', function (event) { if (event.target === cy) showDetailPlaceholder(); });
}

function showDetailPlaceholder() {
  byId('detail-placeholder').style.display = 'flex';
  byId('detail-content').style.display = 'none';
}

function detailHeading(text) {
  var heading = document.createElement('h3');
  heading.textContent = text;
  return heading;
}

function labeledInput(labelText, input) {
  var label = document.createElement('label');
  label.className = 'editor-field';
  var title = document.createElement('span');
  title.textContent = labelText;
  label.append(title, input);
  return label;
}

function button(text, className, handler) {
  var element = document.createElement('button');
  element.type = 'button';
  element.className = className || '';
  element.textContent = text;
  element.onclick = handler;
  return element;
}

function firstTaskFrame(primitive) {
  var entries = Object.entries(primitive.task_frames || {});
  return entries.length ? entries[0] : ['default', { target: [0, 0, 0, 0, 0, 0], learnable_axes: [] }];
}

function showNodeDetail(data) {
  var primitive = data.primitive;
  var roles = primitive.roles || {};
  var frameEntry = firstTaskFrame(primitive);
  var panel = byId('detail-content');
  panel.replaceChildren();
  panel.appendChild(detailHeading('原语：' + primitive.name));

  var meta = document.createElement('p');
  meta.className = 'detail-meta';
  meta.textContent = primitive.type + ' · ' + (data.adaptive ? 'RL' : '脚本') + (roles.is_start ? ' · 起始' : '') + (roles.is_reset ? ' · 复位' : '');
  panel.appendChild(meta);

  var notes = document.createElement('textarea');
  notes.rows = 3;
  notes.value = primitive.notes || '';
  panel.appendChild(labeledInput('说明', notes));

  var terminal = document.createElement('input');
  terminal.type = 'checkbox';
  terminal.checked = Boolean(roles.is_terminal);
  panel.appendChild(labeledInput('终止节点', terminal));

  var axes = ['x', 'y', 'z', 'wx', 'wy', 'wz'];
  var poseGrid = document.createElement('div');
  poseGrid.className = 'pose-editor';
  axes.forEach(function (axis, index) {
    var input = document.createElement('input');
    input.type = 'number';
    input.step = '0.001';
    input.value = frameEntry[1].target[index];
    input.dataset.axis = axis;
    poseGrid.appendChild(labeledInput(axis, input));
  });
  panel.appendChild(labeledInput('目标位姿 (' + frameEntry[0] + ')', poseGrid));

  var learnable = document.createElement('input');
  learnable.type = 'text';
  learnable.value = (frameEntry[1].learnable_axes || []).join(',');
  learnable.placeholder = '例如 x,y,z；留空表示固定';
  panel.appendChild(labeledInput('可学习轴', learnable));

  var actions = document.createElement('div');
  actions.className = 'detail-actions';
  actions.append(
    button('保存属性', 'btn-primary', async function () {
      var rawPrimitive = currentRaw.primitives[primitive.name];
      var rawFrame = rawPrimitive.task_frame.target ? rawPrimitive.task_frame : rawPrimitive.task_frame[frameEntry[0]];
      rawPrimitive.notes = notes.value || null;
      rawPrimitive.is_terminal = terminal.checked;
      rawFrame.target = Array.from(poseGrid.querySelectorAll('input')).map(function (input) { return Number(input.value); });
      var axisNames = learnable.value.split(',').map(function (value) { return value.trim(); }).filter(Boolean);
      var axisOrder = ['x', 'y', 'z', 'wx', 'wy', 'wz'];
      var invalidAxes = axisNames.filter(function (name) { return axisOrder.indexOf(name) < 0; });
      if (invalidAxes.length) { showToast('未知轴：' + invalidAxes.join(', '), 'error'); return; }
      rawFrame.policy_mode = axisOrder.map(function (name) { return axisNames.indexOf(name) >= 0 ? 2 : null; });
      await saveCurrentConfig();
    }),
    button('删除原语', 'btn-danger', function () { deletePrimitive(primitive.name); })
  );
  panel.appendChild(actions);
  byId('detail-placeholder').style.display = 'none';
  panel.style.display = 'block';
}

function showEdgeDetail(transition) {
  var panel = byId('detail-content');
  panel.replaceChildren();
  panel.appendChild(detailHeading('转移 #' + transition.index));
  var meta = document.createElement('p');
  meta.className = 'detail-meta';
  meta.textContent = transition.source + ' → ' + transition.target + '\n' + transition.condition_summary;
  panel.appendChild(meta);

  var source = document.createElement('select');
  var target = document.createElement('select');
  fillPrimitiveSelect(source);
  fillPrimitiveSelect(target);
  source.value = transition.source;
  target.value = transition.target;
  panel.appendChild(labeledInput('起始原语', source));
  panel.appendChild(labeledInput('目标原语', target));

  var type = document.createElement('select');
  Object.keys(transitionTypeMeta).forEach(function (name) {
    type.appendChild(option(name, name));
  });
  type.value = transition.type;
  panel.appendChild(labeledInput('转移类型', type));

  var help = document.createElement('p');
  help.className = 'form-help';
  panel.appendChild(help);
  var parameters = document.createElement('div');
  parameters.className = 'transition-parameter-fields';
  panel.appendChild(parameters);
  renderTransitionParameterFields(parameters, type.value, transition.parameters || {}, help);
  type.onchange = function () {
    renderTransitionParameterFields(parameters, type.value, {}, help);
  };

  var actions = document.createElement('div');
  actions.className = 'detail-actions';
  actions.append(
    button('保存转移', 'btn-primary', async function () {
      currentRaw.transitions[transition.index] = Object.assign({
        source: source.value,
        target: target.value,
        type: type.value
      }, collectTransitionParameters(parameters));
      await saveCurrentConfig();
    }),
    button('删除转移', 'btn-danger', function () { deleteTransition(transition.index); })
  );
  panel.appendChild(actions);
  byId('detail-placeholder').style.display = 'none';
  panel.style.display = 'block';
}

async function editCurrent(operation, argumentsValue) {
  var data = await api('/api/configs/' + encodeURIComponent(currentConfigName) + '/edit', {
    method: 'POST', body: JSON.stringify({ operation: operation, arguments: argumentsValue })
  });
  setLoaded(data);
  return data;
}

async function deletePrimitive(name) {
  if (!window.confirm('确认删除原语“' + name + '”及关联转移？')) return;
  await editCurrent('remove_primitive', { name: name });
  showToast('已删除原语：' + name, 'success');
}

async function deleteTransition(index) {
  if (!window.confirm('确认删除转移 #' + index + '？')) return;
  await editCurrent('remove_transition', { index: index });
  showToast('已删除转移', 'success');
}

async function saveCurrentConfig() {
  if (!currentConfigName || !currentRaw) { showToast('请先选择配置', 'error'); return; }
  syncGlobalSettings();
  var data = await api('/api/configs/' + encodeURIComponent(currentConfigName), {
    method: 'PUT', body: JSON.stringify(currentRaw)
  });
  setLoaded(data);
  await loadConfigList();
  showToast('配置已保存', 'success');
}

async function validateCurrentConfig() {
  if (!currentRaw) { showToast('请先选择配置', 'error'); return; }
  syncGlobalSettings();
  var result = await api('/api/validate', { method: 'POST', body: JSON.stringify(currentRaw) });
  showToast(result.message, 'success');
}

function hideModals() { byId('modal-overlay').style.display = 'none'; }
function showModal(id) {
  byId('modal-overlay').style.display = 'flex';
  byId('modal-add-primitive').style.display = id === 'modal-add-primitive' ? 'block' : 'none';
  byId('modal-add-transition').style.display = id === 'modal-add-transition' ? 'block' : 'none';
}

function fillPrimitiveSelect(select, emptyLabel) {
  select.replaceChildren();
  if (emptyLabel) select.appendChild(option('', emptyLabel));
  (currentSummary ? currentSummary.primitives : []).forEach(function (primitive) { select.appendChild(option(primitive.name)); });
}

function showAddPrimitiveModal() {
  if (!currentRaw) { showToast('请先选择配置', 'error'); return; }
  byId('modal-prim-name').value = '';
  byId('modal-prim-notes').value = '';
  byId('modal-prim-terminal').checked = true;
  fillPrimitiveSelect(byId('modal-prim-connect-from'));
  fillPrimitiveSelect(byId('modal-prim-connect-to'), '不创建出边');
  for (var index = 0; index < 6; index += 1) byId('modal-t' + index).value = 0;
  showModal('modal-add-primitive');
}

function makePrimitive(type, target, notes, terminal) {
  var primitive = {
    type: type,
    task_frame: { target: target, space: 1, policy_mode: [null, null, null, null, null, null], control_mode: [0, 0, 0, 0, 0, 0], origin: [0, 0, 0, 0, 0, 0], min_pose: null, max_pose: null, controller_overrides: null },
    policy_overwrites: {}, notes: notes || null, is_terminal: terminal,
    target_pose_info_key: 'primitive_target_pose'
  };
  if (type === 'move_delta') { primitive.delta = [0, 0, 0, 0, 0, 0]; primitive.delta_frame = 'world'; primitive.publish_target_info = true; }
  if (type === 'open_loop_trajectory') { primitive.trajectory = { target: null, delta: [0, 0, 0, 0, 0, 0], frame: 'task', duration_s: 1 }; primitive.publish_target_info = true; }
  return primitive;
}

async function addPrimitive() {
  var name = byId('modal-prim-name').value.trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)) { showToast('原语名只能使用字母、数字、下划线和连字符', 'error'); return; }
  if (currentRaw.primitives[name]) { showToast('原语已存在：' + name, 'error'); return; }
  var terminal = byId('modal-prim-terminal').checked;
  var connectTo = byId('modal-prim-connect-to').value;
  if (!terminal && !connectTo) { showToast('非终止原语必须选择一个“连接到”节点，避免死胡同', 'error'); return; }
  var target = [];
  for (var index = 0; index < 6; index += 1) target.push(Number(byId('modal-t' + index).value));
  var candidate = deepCopy(currentRaw);
  candidate.primitives[name] = makePrimitive(byId('modal-prim-type').value, target, byId('modal-prim-notes').value.trim(), terminal);
  var connectFrom = byId('modal-prim-connect-from').value;
  if (connectFrom) candidate.transitions.push({ type: 'always', source: connectFrom, target: name, additional_reward: 0, reason: null });
  if (connectTo) candidate.transitions.push({ type: 'always', source: name, target: connectTo, additional_reward: 0, reason: null });
  currentRaw = candidate;
  await saveCurrentConfig();
  hideModals();
}

function showAddTransitionModal() {
  if (!currentRaw) { showToast('请先选择配置', 'error'); return; }
  fillPrimitiveSelect(byId('modal-trans-source'));
  fillPrimitiveSelect(byId('modal-trans-target'));
  updateTransitionParams();
  showModal('modal-add-transition');
}

var TRANSITION_FIELD_LABELS = {
  axes: '检测哪些轴',
  robot_name: '机械臂',
  target_key: '目标位姿信息键',
  tolerance: '到达容差',
  max_steps: '最大步数',
  step_key: '步数信息键',
  event_key: '事件键',
  success_key: '成功标志键',
  obs_key: '观测键',
  operator: '比较方式',
  threshold: '触发阈值',
  pretrained_path: '分类器模型路径',
  device: '推理设备',
  image_size: '图像尺寸',
  metric_key: '指标名称（可选）',
  prob_info_key: '概率信息键'
};

var TRANSITION_FIELD_CHOICES = {
  axes: [
    ['default', '使用原语默认控制轴（推荐）'],
    ['xyz', '位置轴 x、y、z'],
    ['xy', '平面位置轴 x、y'],
    ['z', '仅高度轴 z'],
    ['all', '全部 6 轴']
  ],
  operator: [
    ['ge', '大于等于（≥）'], ['gt', '大于（>）'],
    ['le', '小于等于（≤）'], ['lt', '小于（<）'],
    ['eq', '等于（=）'], ['ne', '不等于（≠）']
  ],
  device: [['cuda', 'CUDA / GPU'], ['cpu', 'CPU']],
  image_size: [['64', '64 × 64'], ['128', '128 × 128（推荐）'], ['224', '224 × 224'], ['256', '256 × 256']],
  step_key: [['step', 'step（当前原语步数）'], ['primitive_step', 'primitive_step'], ['episode_step', 'episode_step（整回合）']],
  target_key: [['primitive_target_pose', 'primitive_target_pose（推荐）']],
  success_key: [['primitive_complete', 'primitive_complete（原语完成）'], ['success', 'success（任务成功）']],
  prob_info_key: [['reward_classifier_prob', 'reward_classifier_prob（推荐）']]
};

function transitionFieldInput(name, specification) {
  var choices = TRANSITION_FIELD_CHOICES[name];
  var input;
  if (choices) {
    input = document.createElement('select');
    choices.forEach(function (choice) { input.appendChild(option(choice[0], choice[1])); });
    if (name === 'axes') {
      input.value = 'default';
    } else if (specification.default !== null && specification.default !== undefined) {
      input.value = String(specification.default);
    }
  } else {
    input = document.createElement('input');
    var typeName = specification.type || '';
    var isNumber = typeName.indexOf('int') >= 0 || typeName.indexOf('float') >= 0;
    input.type = isNumber ? 'number' : 'text';
    if (isNumber) {
      input.step = typeName.indexOf('int') >= 0 ? '1' : 'any';
      if (name === 'max_steps') input.min = '1';
      if (name === 'threshold' || name === 'tolerance') input.min = '0';
    }
    input.value = specification.default === null || specification.default === undefined ? '' : String(specification.default);
    if (name === 'robot_name') input.placeholder = '留空自动选择单机械臂';
    if (name === 'obs_key') input.placeholder = '例如 observation.state';
    if (name === 'event_key') input.placeholder = '例如 grasp_confirmed';
    if (name === 'pretrained_path') input.placeholder = '模型 checkpoint 路径';
  }
  input.dataset.parameter = name;
  input.dataset.parameterType = specification.type || '';
  input.dataset.nullable = specification.default === null ? 'true' : 'false';
  return input;
}

function updateTransitionParams() {
  var type = byId('modal-trans-type').value;
  var metadata = transitionTypeMeta[type] || { fields: {} };
  var container = byId('modal-trans-params');
  container.replaceChildren();
  byId('modal-trans-help').textContent = metadata.doc || '';
  var fieldNames = Object.keys(metadata.fields || {});
  if (!fieldNames.length) {
    var empty = document.createElement('p');
    empty.className = 'empty-parameters';
    empty.textContent = '此转移类型不需要额外参数。';
    container.appendChild(empty);
    return;
  }
  fieldNames.forEach(function (name) {
    container.appendChild(labeledInput(
      TRANSITION_FIELD_LABELS[name] || name,
      transitionFieldInput(name, metadata.fields[name])
    ));
  });
}

function transitionAxesChoice(value) {
  if (value === null || value === undefined) return 'default';
  var serialized = JSON.stringify(value);
  var known = {
    '["x","y","z"]': 'xyz',
    '["x","y"]': 'xy',
    '["z"]': 'z',
    '["x","y","z","wx","wy","wz"]': 'all'
  };
  return known[serialized] || serialized;
}

function setTransitionFieldValue(input, name, value) {
  if (value === undefined) return;
  var selected = name === 'axes' ? transitionAxesChoice(value) : value;
  var serialized = selected === null || selected === undefined ? '' : String(selected);
  if (input.tagName === 'SELECT' && !Array.from(input.options).some(function (item) {
    return item.value === serialized;
  })) {
    input.appendChild(option(serialized, '当前自定义值：' + serialized));
  }
  input.value = serialized;
}

function renderTransitionParameterFields(container, type, currentValues, helpElement) {
  var metadata = transitionTypeMeta[type] || { fields: {} };
  container.replaceChildren();
  helpElement.textContent = metadata.doc || '';
  var fieldNames = Object.keys(metadata.fields || {});
  if (!fieldNames.length) {
    var empty = document.createElement('p');
    empty.className = 'empty-parameters';
    empty.textContent = '此转移类型不需要额外参数。';
    container.appendChild(empty);
    return;
  }
  fieldNames.forEach(function (name) {
    var input = transitionFieldInput(name, metadata.fields[name]);
    setTransitionFieldValue(input, name, currentValues[name]);
    container.appendChild(labeledInput(TRANSITION_FIELD_LABELS[name] || name, input));
  });
}

function collectTransitionParameters(container) {
  var parameters = {};
  (container || byId('modal-trans-params')).querySelectorAll('[data-parameter]').forEach(function (input) {
    var name = input.dataset.parameter;
    var value = input.value;
    if (name === 'axes') {
      var axesByChoice = {
        default: null,
        xyz: ['x', 'y', 'z'],
        xy: ['x', 'y'],
        z: ['z'],
        all: ['x', 'y', 'z', 'wx', 'wy', 'wz']
      };
      parameters[name] = Object.prototype.hasOwnProperty.call(axesByChoice, value)
        ? axesByChoice[value]
        : JSON.parse(value);
      return;
    }
    if (value === '' && input.dataset.nullable === 'true') {
      parameters[name] = null;
      return;
    }
    var typeName = input.dataset.parameterType || '';
    if (typeName.indexOf('int') >= 0) parameters[name] = parseInt(value, 10);
    else if (typeName.indexOf('float') >= 0) parameters[name] = Number(value);
    else parameters[name] = value;
  });
  return parameters;
}

async function addTransition() {
  var parameters = collectTransitionParameters();
  await editCurrent('add_transition', {
    source: byId('modal-trans-source').value,
    target: byId('modal-trans-target').value,
    transition_type: byId('modal-trans-type').value,
    parameters: parameters
  });
  hideModals();
  showToast('已添加转移', 'success');
}

async function initialize() {
  byId('btn-save').onclick = saveCurrentConfig;
  byId('btn-validate').onclick = validateCurrentConfig;
  byId('btn-relayout').onclick = function () { if (cy) cy.layout({ name: typeof window.dagre === 'object' ? 'dagre' : 'breadthfirst', rankDir: 'LR' }).run(); };
  byId('btn-add-primitive').onclick = showAddPrimitiveModal;
  byId('btn-add-transition').onclick = showAddTransitionModal;
  byId('modal-prim-ok').onclick = addPrimitive;
  byId('modal-trans-ok').onclick = addTransition;
  byId('modal-prim-cancel').onclick = hideModals;
  byId('modal-trans-cancel').onclick = hideModals;
  byId('modal-overlay').onclick = function (event) { if (event.target === byId('modal-overlay')) hideModals(); };
  byId('modal-trans-type').onchange = updateTransitionParams;
  byId('btn-new-config').onclick = async function () {
    var name = byId('new-config-name').value.trim();
    if (!name) { showToast('请输入配置名称', 'error'); return; }
    var data = await api('/api/configs', { method: 'POST', body: JSON.stringify({ name: name }) });
    byId('new-config-name').value = '';
    setLoaded(data);
    await loadConfigList();
  };
  try {
    var metadata = await Promise.all([api('/api/transition-types'), api('/api/primitive-types')]);
    transitionTypeMeta = metadata[0];
    primitiveTypeMeta = metadata[1];
    var transitionSelect = byId('modal-trans-type');
    transitionSelect.replaceChildren();
    Object.keys(transitionTypeMeta).forEach(function (name) { transitionSelect.appendChild(option(name)); });
    var primitiveSelect = byId('modal-prim-type');
    primitiveSelect.replaceChildren();
    Object.keys(primitiveTypeMeta).forEach(function (name) { primitiveSelect.appendChild(option(name, name + ' — ' + primitiveTypeMeta[name].doc)); });
    var names = await loadConfigList();
    if (names.length) await loadConfig(names[0]);
  } catch (error) {
    showToast('初始化失败：' + error.message, 'error');
  }
}

document.addEventListener('DOMContentLoaded', initialize);
