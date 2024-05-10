module.exports = {
  apps : [{
    name: 'prompts_generator',
    script: 'propmpt_generator_server.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--arg --arg --arg'
  }]
};