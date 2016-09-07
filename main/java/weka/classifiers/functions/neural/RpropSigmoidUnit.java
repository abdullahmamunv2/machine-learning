/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    SigmoidUnit.java
 *    Copyright (C) 2001-2012 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions.neural;

import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

/**
 * This can be used by the 
 * neuralnode to perform all it's computations (as a sigmoid unit).
 *
 * @author Malcolm Ware (mfw4@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class RpropSigmoidUnit
  implements RpropNeuralMethod, RevisionHandler {

  /** for serialization */
  private static final long serialVersionUID = -5162958458177475652L;
  
  /**
   * This function calculates what the output value should be.
   * @param node The node to calculate the value for.
   * @return The value.
   */
  public double outputValue(RpropNeuralNode node) {
    double[] weights = node.getWeights();
    RpropNeuralConnection[] inputs = node.getInputs();
    double value = weights[0];
    for (int noa = 0; noa < node.getNumInputs(); noa++) {
      
      value += inputs[noa].outputValue(true) 
	* weights[noa+1];
    }
     
    //this I got from the Neural Network faq to combat overflow
    //pretty simple solution really :)
    if (value < -45) {
      value = 0;
    }
    else if (value > 45) {
      value = 1;
    }
    else {
      value = 1 / (1 + Math.exp(-value));
    }  
    return value;
  }
  
  /**
   * This function calculates what the error value should be.
   * @param node The node to calculate the error for.
   * @return The error.
   */
  public double errorValue(RpropNeuralNode node) {
    //then calculate the error.
    
    RpropNeuralConnection[] outputs = node.getOutputs();
    int[] oNums = node.getOutputNums();
    double error = 0;
    
    for (int noa = 0; noa < node.getNumOutputs(); noa++) {
      error += outputs[noa].errorValue(true) 
	* outputs[noa].weightValue(oNums[noa]);
    }
    double value = node.outputValue(false);
    error *= value * (1 - value);
    
    return error;
  }

  /**
   * This function will calculate what the change in weights should be
   * and also update them.
   * @param node The node to update the weights for.
   * @param learn The learning rate to use.
   * @param momentum The momentum to use.
   */
  public void updateWeights(RpropNeuralNode node, double positiveEta, double negativeEta) {

    RpropNeuralConnection[] inputs = node.getInputs();
    double[] cWeights = node.getChangeInWeights();
    double[] weights = node.getWeights();
    double[] deltaUpdate=node.getDeltaUpdate();
    double [] previousError=node.getPreviousError();
    
    
    double errorValue=node.errorValue(false);
    int stopValue = node.getNumInputs() + 1;
    for (int noa = 1; noa < stopValue; noa++) {
        double currentError=inputs[noa-1].outputValue(false)*errorValue;
        double c=0;
        if(currentError*previousError[noa] > 0 ){
            double delta=Math.min(deltaUpdate[noa]*positiveEta, node.getDeltaMax());
            c=((-1)*(int) Math.signum(previousError[noa]))*delta;
            weights[noa]+=c;
            cWeights[noa]=c;
            deltaUpdate[noa]=delta;
            previousError[noa]=currentError;
        }
        else if(currentError*previousError[noa] < 0){
            double delta=Math.max(deltaUpdate[noa]*negativeEta, node.getDeltaMin());
            weights[noa]-=cWeights[noa];
            deltaUpdate[noa]=delta;
            previousError[noa]=0;
        }
        else{
            c=((-1)*(int) Math.signum(previousError[noa]))*deltaUpdate[noa];
            weights[noa]+=c;
            cWeights[noa]=c;
        }
            
    }
    

  }
  
  /*public void updateWeights(NeuralNode node, double learn, double momentum) {

    NeuralConnection[] inputs = node.getInputs();
    double[] cWeights = node.getChangeInWeights();
    double[] weights = node.getWeights();
    double learnTimesError = 0;
    learnTimesError = learn * node.errorValue(false);  
    
  
    // c= delta Wij(t)
    // error rate dE/dWij=node.errorValue(false);
    // delta Wij(t-1)=cWeights[0]
      
    double c = learnTimesError + momentum * cWeights[0];     
    weights[0] += c;
    cWeights[0] = c;
 
    int stopValue = node.getNumInputs() + 1;
    for (int noa = 1; noa < stopValue; noa++) {
      
      c = learnTimesError * inputs[noa-1].outputValue(false);
      c += momentum * cWeights[noa];
      
      weights[noa] += c;
      cWeights[noa] = c; 
    }
  }*/
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }

}
