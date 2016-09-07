/*
 * $Id: NodeTreeNode.java,v 1.2 2008/05/20 20:19:20 edankert Exp $
 *
 * Copyright (c) 2002 - 2008, Edwin Dankert
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright 
 *   notice, this list of conditions and the following disclaimer in the 
 *   documentation and/or other materials provided with the distribution. 
 * * Neither the name of 'Edwin Dankert' nor the names of its contributors 
 *   may  be used to endorse or promote products derived from this software 
 *   without specific prior written permission. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR 
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package org.bounce.viewer.xml;

import java.util.ArrayList;
import java.util.List;

import javax.swing.tree.DefaultMutableTreeNode;

import org.w3c.dom.Node;

/**
 * The node for the XML tree, containing an XML element.
 *
 * @author Edwin Dankert <edankert@gmail.com>
 */
public abstract class NodeTreeNode extends DefaultMutableTreeNode {
	private static final long serialVersionUID = 2859132085591886595L;
    protected static final int MAX_LINE_LENGTH = 80;
	private List<Line> lines = null;
	private Line current = null; 
	
	/**
	 * Constructs the node for the XML element.
	 *
	 * @param element the XML element.
	 */	
	NodeTreeNode(Node node) {
		super(node);
	}
	
	void update() {
		for ( int i = 0; i < getChildCount(); i++) {
			NodeTreeNode node = (NodeTreeNode)getChildAt( i);
			node.update();
		}
		
		format();
	}
	
	/**
	 * Returns the formatted lines for this element.
	 *
	 * @return the formatted Lines.
	 */	
	List<Line> getLines() {
		if (lines == null) {
			lines = new ArrayList<Line>();
		}
		
		return lines;
	}
	
	/**
	 * Constructs the node for the XML element.
	 *
	 * @param element the XML element.
	 */	
	public Node getNode() {
		return (Node)getUserObject();
	}
	
	abstract void format();
	
	void setCurrent(Line current) {
		this.current = current;

		getLines().add(current);
	}
	
	Line getCurrent() {
		return current;
	}
} 
