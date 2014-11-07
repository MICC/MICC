# -*- mode: python -*-
a = Analysis(['micc/cli.py'],
             pathex=['/home/mattt/Code/Git/MICC'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='MICC-linux',
          debug=False,
          strip=None,
          upx=True,
          console=True )
