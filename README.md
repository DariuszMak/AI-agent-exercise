# Python AI Agent Excercise

### Project structure diagrams

##### Modular perspective

<p align="center">
  <img src="images/structure_module.svg" alt="Modular perspective" width="600">
</p>

##### Library dependencies perspective

<p align="center">
  <img src="images/structure_module_clustered.svg" alt="Library dependencies perspective" width="600">
</p>

## Requirements

- [UV](https://github.com/astral-sh/uv) package manager
- [Task](https://taskfile.dev/docs/installation) runner

## Local development (Windows PowerShell)

You can also use VSCode `settings.json` and `launch.json` files to run the project (choose interpreter created by UV).

### Fast Windows dev

```commandline
task full-dev-windows ; 
```

### Full analysis

```commandline
task full-static-analyzis ; 
```

Check installed models:

```commandline
ollama list ; 
```