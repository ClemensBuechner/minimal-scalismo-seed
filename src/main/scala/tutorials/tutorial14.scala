package tutorials

import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}

object tutorial14 {

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val mu = -5
    val sigma = 17

    val trueDistribution = breeze.stats.distributions.Gaussian(mu, sigma)
    val data = for (_ <- 0 until 100) yield {
      trueDistribution.draw()
    }

    case class Parameters(mu: Double, sigma: Double)
    case class Sample(parameters: Parameters, generatedBy: String)

    case class LikelihoodEvaluator(data: Seq[Double]) extends DistributionEvaluator[Sample] {

      override def logValue(theta: Sample): Double = {
        val likelihood = breeze.stats.distributions.Gaussian(
          theta.parameters.mu, theta.parameters.sigma
        )
        val likelihoods = for (x <- data) yield {
          likelihood.logPdf(x)
        }
        likelihoods.sum
      }
    }

    object PriorEvaluator extends DistributionEvaluator[Sample] {

      val priorDistMu = breeze.stats.distributions.Gaussian(0, 20)
      val priorDistSigma = breeze.stats.distributions.Gaussian(0, 100)

      override def logValue(theta: Sample): Double = {
        priorDistMu.logPdf(theta.parameters.mu)
        +priorDistSigma.logPdf(theta.parameters.sigma)
      }
    }

    val posteriorEvaluator = ProductEvaluator(PriorEvaluator, LikelihoodEvaluator(data))

    case class RandomWalkProposal(stddevMu: Double, stddevSigma: Double)(implicit rng: scalismo
    .utils.Random) extends ProposalGenerator[Sample] with TransitionProbability[Sample] {

      override def propose(sample: Sample): Sample = {
        val newParameters = Parameters(
          mu = sample.parameters.mu + rng.breezeRandBasis.gaussian(0, stddevMu).draw(),
          sigma = sample.parameters.sigma + rng.breezeRandBasis.gaussian(0, stddevSigma).draw()
        )

        Sample(newParameters, s"randomWalkProposal ($stddevMu, $stddevSigma)")
      }

      override def logTransitionProbability(from: Sample, to: Sample): Double = {

        val stepDistMu = breeze.stats.distributions.Gaussian(0, stddevMu)
        val stepDistSigma = breeze.stats.distributions.Gaussian(0, stddevSigma)

        val residualMu = to.parameters.mu - from.parameters.mu
        val residualSigma = to.parameters.sigma - from.parameters.sigma
        stepDistMu.logPdf(residualMu) + stepDistMu.logPdf(residualSigma)
      }
    }

    val smallStepProposal = RandomWalkProposal(3.0, 1.0)
    val largeStepProposal = RandomWalkProposal(9.0, 3.0)

    val generator = MixtureProposal.fromProposalsWithTransition[Sample](
      (0.8, smallStepProposal),
      (0.2, largeStepProposal)
    )

    val chain = MetropolisHastings(generator, posteriorEvaluator)

    val initialSample = Sample(Parameters(0.0, 10.0), generatedBy = "initial")
    val mhIterator = chain.iterator(initialSample)

    // stop
    val samples = mhIterator.drop(1000).take(5000).toIndexedSeq

    val estimatedMean = samples.map(sample => sample.parameters.mu).sum / samples.size
    // estimatedMean: Double = -5.791574550006766
    println("estimated mean is " + estimatedMean)
    // estimated mean is -5.791574550006766

    val estimatedSigma = samples.map(sample => sample.parameters.sigma).sum / samples.size
    // estimatedSigma: Double = 17.350744030639415
    println("estimated sigma is " + estimatedSigma)
    // estimated sigma is 17.350744030639415

    class Logger extends AcceptRejectLogger[Sample] {
      private val numAccepted = collection.mutable.Map[String, Int]()
      private val numRejected = collection.mutable.Map[String, Int]()

      override def accept(current: Sample,
                          sample: Sample,
                          generator: ProposalGenerator[Sample],
                          evaluator: DistributionEvaluator[Sample]
                         ): Unit = {
        val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
        numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
      }

      override def reject(current: Sample,
                          sample: Sample,
                          generator: ProposalGenerator[Sample],
                          evaluator: DistributionEvaluator[Sample]
                         ): Unit = {
        val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
        numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
      }


      def acceptanceRatios(): Map[String, Double] = {
        val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
        val acceptanceRatios = for (generatorName <- generatorNames) yield {
          val total = (numAccepted.getOrElse(generatorName, 0)
            + numRejected.getOrElse(generatorName, 0)).toDouble
          (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
        }
        acceptanceRatios.toMap
      }
    }

    val logger = new Logger()
    val mhIteratorWithLogging = chain.iterator(initialSample, logger)
    val samples2 = mhIteratorWithLogging.drop(1000).take(3000).toIndexedSeq

    println("acceptance ratio is " + logger.acceptanceRatios())
    // acceptance ratio is Map(randomWalkProposal (3.0, 1.0) -> 0.45588235294117646,
    // randomWalkProposal (9.0, 3.0) -> 0.1382316313823163)
  }
}

