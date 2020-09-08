// <copyright file="NNISweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

namespace MLNet.Sweeper
{
    public class NNISweeper : ISweeper
    {
        public const string SearchSpaceJson = "parameter.cfg";

        public void AddRunHistory(IRunResult input)
        {
            throw new NotImplementedException();
        }

        public object Clone()
        {
            throw new NotImplementedException();
        }

        public IEnumerable<IDictionary<string, string>> ProposeSweeps(ISweepable sweepingSpace, int maxSweeps = 100, IEnumerable<IRunResult> previousRuns = null)
        {
            using (var stream = new StreamReader(SearchSpaceJson))
            {
                var json = stream.ReadToEnd();
                var parameters = JsonConvert.DeserializeObject<Dictionary<string, string>>(json);
                yield return parameters;
            }
        }
    }
}
