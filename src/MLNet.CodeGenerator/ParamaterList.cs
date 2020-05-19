using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.CodeGenerator
{
    public class ParamaterList : List<Paramater>, ICodeGenNode
    {
        public string GeneratorCode()
        {
            return string.Join(",", this.Select(x => x.GeneratorCode()));
        }
    }
}
