"""Prepare a single shared FastScan implementation without changing source trees."""
import difflib
from pathlib import Path

from diagnose_succinct import OUT, SUCCINCT


def prepare():
    relative=Path('src/turbo/distance/avx512/symqg/symqg_kernels_avx512.cc')
    original=(SUCCINCT/relative).read_text()
    assert 'symqg_scan_impl' not in original, 'Compact implementation is already present'
    start=original.index('bool symqg_scan(')
    end=original.index('\n}  // namespace zvec::turbo',start)
    function=original[start:end]
    signature=function[:function.index(' {')]
    body=function[function.index(' {'):]
    gate='\n  if (!symqg_avx512_available() || padded_dim == 0 || (padded_dim & 15) != 0)\n    return false;\n'
    assert gate in body
    body=body.replace(gate,'\n  const size_t padded_dim = Fixed ? Fixed : dimension;\n',1)
    loop='    for (size_t offset = dim * 4; offset < (dim + slice) * 4; offset += 64) {'
    assert loop in body
    body=body.replace(loop,'#pragma GCC unroll 4\n'+loop,1)
    implementation=('// Keep one arithmetic implementation for specialized and generic scans.\n'
                    '// Fixed == 0 preserves support for all other valid dimensions.\n'
                    'template <size_t Fixed>\nstatic '+
                    signature.replace('symqg_scan(', 'symqg_scan_impl(',1).replace('size_t padded_dim','size_t dimension')+
                    body)
    dispatch=(signature+' {'+gate+'''
  const auto scan = [&](auto fixed) {
    return symqg_scan_impl<decltype(fixed)::value>(
        codes, lut, padded_dim, add, scale, center_distance, delta, sum_vl,
        k1xsumq, distances);
  };
  switch (padded_dim) {
    case 64: return scan(std::integral_constant<size_t, 64>{});
    case 128: return scan(std::integral_constant<size_t, 128>{});
    case 256: return scan(std::integral_constant<size_t, 256>{});
    case 512: return scan(std::integral_constant<size_t, 512>{});
    case 1024: return scan(std::integral_constant<size_t, 1024>{});
    default: return scan(std::integral_constant<size_t, 0>{});
  }
}
''')
    changed=original[:start]+implementation+'\n'+dispatch+original[end:]
    changed=changed.replace('#include <cmath>','#include <cmath>\n#include <type_traits>',1)
    folder=OUT/'compact-source'
    folder.mkdir(exist_ok=True)
    target=folder/'symqg_kernels_avx512.cc'
    target.write_text(changed)
    (folder/'kernel.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),
        changed.splitlines(True),fromfile='a/'+str(relative),tofile='b/'+str(relative))))
    print(target)
    return target


if __name__=='__main__':
    prepare()
