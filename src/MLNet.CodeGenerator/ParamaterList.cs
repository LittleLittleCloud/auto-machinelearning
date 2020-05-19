using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.CodeGenerator
{
    public class ParamaterList : List<Paramater>, ICodeGenNode
    {
        public string GeneratorCode()
        {
            return string.Join(",", this);
        }
    }
}
