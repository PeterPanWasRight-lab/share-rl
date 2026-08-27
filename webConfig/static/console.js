'use strict';

var activeView = 'overview';
var currentProfile = null;
var refreshTimer = null;
var viewerExamplesLoaded = false;

function byId(id) { return document.getElementById(id); }

async function api(url, options) {
  var response = await fetch(url, Object.assign({
    headers: { 'Content-Type': 'application/json' },
    cache: 'no-store'
  }, options || {}));
  var text = await response.text();
  var payload = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(payload.error || ('HTTP ' + response.status));
  return payload;
}

function toast(message, type) {
  var element = byId('toast');
  element.textContent = message;
  element.className = 'toast show ' + (type || '');
  window.setTimeout(function () { element.className = 'toast'; }, 3600);
}

function formatNumber(value) {
  if (value === null || value === undefined) return '—';
  return Number(value).toLocaleString('zh-CN');
}

function titleFor(view) {
  return {
    overview: '实验总览', services: 'Actor / Learner', buffer: 'Replay Buffer',
    'state-machine': 'MP-Net 状态机', algorithm: '算法参数'
  }[view] || view;
}

function navigate(view) {
  activeView = view;
  document.querySelectorAll('.view').forEach(function (element) {
    element.classList.toggle('active', element.id === 'view-' + view);
  });
  document.querySelectorAll('.nav-item').forEach(function (element) {
    element.classList.toggle('active', element.dataset.view === view);
  });
  byId('page-title').textContent = titleFor(view);
  if (view === 'buffer') refreshReplay();
  if (view === 'services') refreshServices(true);
  if (view === 'state-machine') refreshViewerRun();
}

function serviceLabel(status) {
  if (status.state === 'running') return '运行中';
  if (status.state === 'exited') return '已退出';
  return '已停止';
}

function renderServices(services) {
  var running = Object.values(services).filter(function (item) { return item.state === 'running'; }).length;
  byId('stat-services').textContent = running + ' / 2';
  byId('overview-services').innerHTML = ['learner', 'actor'].map(function (role) {
    var item = services[role];
    return '<div class="mini-service"><div><b>' + (role === 'learner' ? 'Learner' : 'Actor') +
      '</b><span>' + (item.pid ? 'PID ' + item.pid + ' · ' + (item.uptime_s || 0) + 's' : '尚未由控制台启动') +
      '</span></div><span class="status-pill ' + item.state + '">' + serviceLabel(item) + '</span></div>';
  }).join('');

  ['learner', 'actor'].forEach(function (role) {
    var item = services[role];
    var card = document.querySelector('.service-card[data-role="' + role + '"]');
    if (!card) return;
    var pill = card.querySelector('.status-pill');
    pill.textContent = serviceLabel(item);
    pill.className = 'status-pill ' + item.state;
    card.querySelector('[data-value="pid"]').textContent = item.pid || '—';
    card.querySelector('[data-value="uptime"]').textContent = item.uptime_s === undefined ? '—' : item.uptime_s + 's';
    card.querySelector('[data-value="exit"]').textContent = item.exit_code === null ? '—' : item.exit_code;
    card.querySelector('.start-service').disabled = item.state === 'running';
    card.querySelector('.stop-service').disabled = item.state !== 'running';
  });
}

function bufferSummary(metrics) {
  var primitives = Object.values((metrics || {}).primitives || {});
  return primitives.reduce(function (total, item) {
    total.online += Number((item.online || {}).transitions || 0);
    total.offline += Number((item.offline || {}).transitions || 0);
    total.updates += Number(item.optimization_step || 0);
    return total;
  }, { online: 0, offline: 0, updates: 0 });
}

function bufferTable(name, data) {
  data = data || {};
  return '<h4>' + name + '</h4><table class="buffer-table"><thead><tr><th>项目</th><th>Transitions</th><th>轨迹</th><th>成功</th><th>失败</th><th>干预</th></tr></thead><tbody><tr><td>' +
    (data.fill_percent || 0) + '%</td><td>' + formatNumber(data.transitions || 0) + '</td><td>' + formatNumber(data.completed_trajectories || 0) +
    '</td><td>' + formatNumber(data.successes || 0) + '</td><td>' + formatNumber(data.terminal_failures || 0) + '</td><td>' + formatNumber(data.interventions || 0) +
    '</td></tr></tbody></table><div class="progress"><i style="width:' + Math.min(100, data.fill_percent || 0) + '%"></i></div>';
}

function renderReplay(result) {
  var connected = Boolean(result && result.connected);
  byId('replay-badge').textContent = connected ? '实时连接' : '等待 Learner';
  byId('replay-badge').className = 'badge' + (connected ? ' connected' : '');
  byId('buffer-connection').textContent = connected ? '已连接 · 每 2 秒刷新' : '未连接';
  var metrics = connected ? result.metrics : { primitives: {} };
  var summary = bufferSummary(metrics);
  byId('stat-online').textContent = connected ? formatNumber(summary.online) : '—';
  byId('stat-updates').textContent = connected ? formatNumber(summary.updates) : '—';
  if (!connected) {
    byId('overview-buffer').className = 'empty-state';
    byId('overview-buffer').textContent = 'Learner 启动后，这里会每秒展示在线与离线数据。';
    byId('buffer-grid').innerHTML = '<article class="panel empty-state">无法连接 ' + (result ? result.url : '') + '<br>请确认 Learner 与 Replay Dashboard 已启动。</article>';
    return;
  }
  byId('overview-buffer').className = '';
  byId('overview-buffer').innerHTML = '<div class="stats-grid"><article class="stat"><span>Online</span><strong>' + formatNumber(summary.online) + '</strong><small>Transitions</small></article><article class="stat"><span>Offline</span><strong>' + formatNumber(summary.offline) + '</strong><small>Transitions</small></article></div>';
  var entries = Object.entries(metrics.primitives || {});
  byId('buffer-grid').innerHTML = entries.length ? entries.map(function (entry) {
    var id = entry[0], item = entry[1];
    return '<article class="panel buffer-card"><h3>' + id + '</h3><p>Optimization step ' + formatNumber(item.optimization_step) + '</p>' + bufferTable('Online buffer', item.online) + bufferTable('Offline buffer', item.offline) + '</article>';
  }).join('') : '<article class="panel empty-state">已连接，但 Learner 尚未创建可学习 primitive 的 Buffer。</article>';
}

async function refreshSummary() {
  try {
    var summary = await api('/api/console/summary');
    byId('server-dot').className = 'connection-dot online';
    byId('server-label').textContent = '本地服务正常';
    byId('stat-configs').textContent = summary.config_count;
    currentProfile = summary.profile;
    byId('profile-chip').textContent = currentProfile.name + ' · ' + currentProfile.device.toUpperCase();
    renderServices(summary.services);
    renderReplay(summary.replay);
  } catch (error) {
    byId('server-dot').className = 'connection-dot';
    byId('server-label').textContent = '后端连接失败';
    toast(error.message, 'error');
  }
}

async function refreshServices(withLogs) {
  try {
    var services = await api('/api/console/services');
    renderServices(services);
    if (withLogs) {
      await Promise.all(['learner', 'actor'].map(async function (role) {
        var result = await api('/api/console/services/' + role + '/log');
        document.querySelector('[data-log="' + role + '"]').textContent = result.text || '尚无日志';
      }));
    }
  } catch (error) { toast(error.message, 'error'); }
}

async function refreshReplay() {
  try { renderReplay(await api('/api/console/replay')); }
  catch (error) { toast(error.message, 'error'); }
}

async function serviceAction(role, action) {
  var verb = action === 'start' ? '启动' : '停止';
  if (!window.confirm('确认' + verb + ' ' + role.toUpperCase() + '？')) return;
  try {
    await api('/api/console/services/' + role + '/' + action, { method: 'POST', body: '{}' });
    toast(role.toUpperCase() + ' 已' + verb, 'success');
    await refreshServices(true);
  } catch (error) { toast(error.message, 'error'); }
}

function profileValue(input, value) {
  if (input.type === 'checkbox') input.checked = Boolean(value);
  else input.value = value === null || value === undefined ? '' : value;
}

async function loadProfile() {
  try {
    currentProfile = await api('/api/console/profile');
    document.querySelectorAll('#profile-form [data-field]').forEach(function (input) {
      profileValue(input, currentProfile[input.dataset.field]);
    });
    byId('profile-chip').textContent = currentProfile.name + ' · ' + currentProfile.device.toUpperCase();
  } catch (error) { toast(error.message, 'error'); }
}

function collectProfile() {
  var next = {};
  document.querySelectorAll('#profile-form [data-field]').forEach(function (input) {
    var value = input.type === 'checkbox' ? input.checked : input.value;
    if (input.type === 'number') value = Number(value);
    next[input.dataset.field] = value;
  });
  return next;
}

async function saveProfile() {
  try {
    currentProfile = await api('/api/console/profile', { method: 'PUT', body: JSON.stringify(collectProfile()) });
    toast('运行参数已保存；下次启动时生效', 'success');
    await loadProfile();
  } catch (error) { toast(error.message, 'error'); }
}

async function previewCommand(role) {
  try {
    var result = await api('/api/console/services/' + role + '/command');
    byId('command-preview').textContent = result.argv.map(function (part) {
      return /\s/.test(part) ? JSON.stringify(part) : part;
    }).join(' \\\n  ');
  } catch (error) { toast(error.message, 'error'); }
}

function currentEditorConfigName() {
  try {
    return byId('editor-frame').contentWindow.currentConfigName || null;
  } catch (_error) {
    return null;
  }
}

function renderViewerRun(status) {
  var running = status.state === 'running';
  var pill = byId('viewer-status');
  pill.textContent = running ? '运行中' : (status.state === 'exited' ? '已退出' : '已停止');
  pill.className = 'status-pill ' + status.state;
  byId('start-viewer-example').disabled = running;
  byId('stop-viewer-example').disabled = !running;
  var configName = running || status.state === 'exited' ? status.config_name : currentEditorConfigName();
  byId('viewer-config').textContent = configName ? ('配置：' + configName) : '请先在下方选择配置';
}

async function refreshViewerRun() {
  try {
    var result = await api('/api/console/examples');
    if (!viewerExamplesLoaded) {
      var select = byId('viewer-example');
      select.replaceChildren();
      result.examples.forEach(function (item) {
        var optionElement = document.createElement('option');
        optionElement.value = item.id;
        optionElement.textContent = item.label + ' · ' + item.script;
        select.appendChild(optionElement);
      });
      viewerExamplesLoaded = true;
    }
    renderViewerRun(result.run);
    var log = await api('/api/console/example-run/log');
    byId('viewer-example-log').textContent = log.text || '尚无日志';
  } catch (error) { toast(error.message, 'error'); }
}

async function startViewerExample() {
  var frameWindow = byId('editor-frame').contentWindow;
  var configName = currentEditorConfigName();
  if (!configName || typeof frameWindow.saveCurrentConfig !== 'function') {
    toast('请先在状态机编辑器中选择配置', 'error');
    return;
  }
  try {
    await frameWindow.saveCurrentConfig();
    await api('/api/console/example-run/start', {
      method: 'POST',
      body: JSON.stringify({
        example_id: byId('viewer-example').value,
        config_name: configName,
        steps: Number(byId('viewer-steps').value)
      })
    });
    toast('配置已保存，Viewer 正在启动', 'success');
    await refreshViewerRun();
  } catch (error) { toast(error.message, 'error'); }
}

async function stopViewerExample() {
  if (!window.confirm('确认停止当前状态机 Viewer 试运行？')) return;
  try {
    await api('/api/console/example-run/stop', { method: 'POST', body: '{}' });
    toast('Viewer 试运行已停止', 'success');
    await refreshViewerRun();
  } catch (error) { toast(error.message, 'error'); }
}

function bindEvents() {
  document.querySelectorAll('.nav-item').forEach(function (button) {
    button.onclick = function () { navigate(button.dataset.view); };
  });
  document.querySelectorAll('[data-jump]').forEach(function (button) {
    button.onclick = function () { navigate(button.dataset.jump); };
  });
  document.querySelectorAll('.start-service').forEach(function (button) {
    button.onclick = function () { serviceAction(button.dataset.role, 'start'); };
  });
  document.querySelectorAll('.stop-service').forEach(function (button) {
    button.onclick = function () { serviceAction(button.dataset.role, 'stop'); };
  });
  document.querySelectorAll('.preview-command').forEach(function (button) {
    button.onclick = function () { previewCommand(button.dataset.role); };
  });
  byId('save-profile').onclick = saveProfile;
  byId('start-viewer-example').onclick = startViewerExample;
  byId('stop-viewer-example').onclick = stopViewerExample;
  byId('refresh-all').onclick = async function () { await refreshSummary(); if (activeView === 'services') await refreshServices(true); };
}

async function initialize() {
  bindEvents();
  await Promise.all([loadProfile(), refreshSummary()]);
  refreshTimer = window.setInterval(function () {
    if (activeView === 'services') refreshServices(true);
    else if (activeView === 'buffer') refreshReplay();
    else if (activeView === 'state-machine') refreshViewerRun();
    else if (activeView === 'overview') refreshSummary();
  }, 2000);
}

document.addEventListener('DOMContentLoaded', initialize);
