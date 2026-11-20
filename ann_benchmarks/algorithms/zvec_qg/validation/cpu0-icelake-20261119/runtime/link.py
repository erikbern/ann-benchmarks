from pathlib import Path
import json,shlex,subprocess,sys,sysconfig
r=Path(__file__).resolve().parent
b=r/'core/build'
args=shlex.split(next(x['command'] for x in json.loads((b/'compile_commands.json').read_text()) if x['file'].endswith('/local_builder.cc')))
args[args.index('-o')+1]=str(r/'direct.o');args[args.index('-c')+1]=str(r/'direct.cpp')
args+=['-fPIC','-fvisibility=hidden']+shlex.split(subprocess.check_output([sys.executable,'-m','pybind11','--includes'],text=True))
link=shlex.split(subprocess.check_output([str(Path(sys.executable).parent/'ninja'),'-C',str(b),'-t','commands','bin/local_builder'],text=True).splitlines()[-1])[2:-2]
link[link.index('-o')+1]=str(r/('_direct_core'+sysconfig.get_config_var('EXT_SUFFIX')))
link=[str(r/'direct.o') if x.endswith('/local_builder.cc.o') else x for x in link if not x.startswith('-Wl,-rpath,')]
link+=['-shared','-Wl,-rpath,'+str(b/'lib')]
(r/'compile-bridge.json').write_text(json.dumps([args,link],indent=2))
subprocess.run(args,cwd=b,check=True)
subprocess.run(link,cwd=b,check=True)
