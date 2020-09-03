using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace MLNet.AutoPipeline.Test
{
    public class IterationInfoTest
    {
        [Theory]
        [InlineData(1, 1, true, false, true, false)]
        [InlineData(1, 0, true, false, false, true)]
        [InlineData(0, 1, true, true, false, false)]
        [InlineData(1, 1, false, false, true, false)]
        [InlineData(1, 0, false, true, false, false)]
        [InlineData(0, 1, false, false, false, true)]
        public void IterationInfo_should_be_comparable(double score1, double score2, bool isMaximizing, bool smaller, bool equal, bool bigger)
        {
            var info1 = new IterationInfo(null, null, 0, score1, isMaximizing);
            var info2 = new IterationInfo(null, null, 0, score2, isMaximizing);

            if (smaller)
            {
                Assert.True(info1 < info2);
            }

            if (equal)
            {
                Assert.True(info1 <= info2 && info1 >= info2);
            }

            if (bigger)
            {
                Assert.True(info1 > info2);
            }
        }
    }
}
