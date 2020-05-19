// <copyright file="EstimatorChain.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace MLNet.CodeGenerator
{
    public class EstimatorChain : List<Estimator>, ICodeGenNode
    {
        public string GeneratorCode()
        {
            if (this.Count < 2)
            {
                throw new Exception("EstimatorChain length must be greater than 1");
            }

            var sb = new StringBuilder();
            sb.AppendLine(this[0].GeneratorCode());

            for (int i = 1; i != this.Count - 1; ++i)
            {
                sb.AppendLine($".Append({this[i].GeneratorCode()})");
            }

            sb.AppendLine($".Append({this[this.Count - 1].GeneratorCode()});");

            var code = sb.ToString();
            return Utils.FormatCode(code);
        }
    }
}
