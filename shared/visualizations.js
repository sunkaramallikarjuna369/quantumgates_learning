/**
 * Quantum Gates Visualization Library
 * Shared utilities for 3D Bloch sphere and circuit visualizations
 */

class BlochSphere {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            width: options.width || this.container.clientWidth,
            height: options.height || 500,
            backgroundColor: options.backgroundColor || 0x0a0a0a,
            sphereColor: options.sphereColor || 0x64ffda,
            ...options
        };
        
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);
        
        this.camera = new THREE.PerspectiveCamera(75, this.options.width / this.options.height, 0.1, 1000);
        this.camera.position.set(2.5, 2, 2.5);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.options.width, this.options.height);
        this.container.appendChild(this.renderer.domElement);
        
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        this.setupLights();
        this.createSphere();
        this.createAxes();
        
        this.stateVectors = [];
        this.animations = [];
        this.time = 0;
        
        this.animate();
        this.setupResize();
    }
    
    setupLights() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0xffffff, 0.8);
        pointLight.position.set(10, 10, 10);
        this.scene.add(pointLight);
    }
    
    createSphere() {
        // Transparent sphere
        const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: this.options.sphereColor,
            transparent: true,
            opacity: 0.15
        });
        this.sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        this.scene.add(this.sphere);
        
        // Wireframe
        const wireframeGeometry = new THREE.SphereGeometry(1, 16, 16);
        const wireframeMaterial = new THREE.MeshBasicMaterial({
            color: this.options.sphereColor,
            wireframe: true,
            transparent: true,
            opacity: 0.2
        });
        this.wireframe = new THREE.Mesh(wireframeGeometry, wireframeMaterial);
        this.scene.add(this.wireframe);
        
        // Equator
        const equatorGeometry = new THREE.TorusGeometry(1, 0.01, 16, 100);
        const equatorMaterial = new THREE.MeshBasicMaterial({ color: 0xffd93d, transparent: true, opacity: 0.4 });
        this.equator = new THREE.Mesh(equatorGeometry, equatorMaterial);
        this.equator.rotation.x = Math.PI / 2;
        this.scene.add(this.equator);
    }
    
    createAxes() {
        const axisLength = 1.4;
        
        // X axis (red)
        this.xAxis = new THREE.ArrowHelper(
            new THREE.Vector3(1, 0, 0),
            new THREE.Vector3(0, 0, 0),
            axisLength, 0xff6b6b, 0.1, 0.05
        );
        this.scene.add(this.xAxis);
        
        // Y axis (green)
        this.yAxis = new THREE.ArrowHelper(
            new THREE.Vector3(0, 1, 0),
            new THREE.Vector3(0, 0, 0),
            axisLength, 0x4ecdc4, 0.1, 0.05
        );
        this.scene.add(this.yAxis);
        
        // Z axis (blue/yellow)
        this.zAxis = new THREE.ArrowHelper(
            new THREE.Vector3(0, 0, 1),
            new THREE.Vector3(0, 0, 0),
            axisLength, 0xffd93d, 0.1, 0.05
        );
        this.scene.add(this.zAxis);
        
        // Axis labels using sprites
        this.addLabel('X', 1.5, 0, 0, '#ff6b6b');
        this.addLabel('Y', 0, 1.5, 0, '#4ecdc4');
        this.addLabel('Z', 0, 0, 1.5, '#ffd93d');
        this.addLabel('|0⟩', 0, 0, 1.2, '#ffffff');
        this.addLabel('|1⟩', 0, 0, -1.2, '#ffffff');
    }
    
    addLabel(text, x, y, z, color) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 64;
        canvas.height = 64;
        
        context.fillStyle = color;
        context.font = 'bold 48px Arial';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, 32, 32);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        sprite.position.set(x, y, z);
        sprite.scale.set(0.3, 0.3, 0.3);
        this.scene.add(sprite);
    }
    
    addStateVector(theta, phi, color = 0xff6b6b, label = '') {
        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.sin(theta) * Math.sin(phi);
        const z = Math.cos(theta);
        
        const arrow = new THREE.ArrowHelper(
            new THREE.Vector3(x, y, z).normalize(),
            new THREE.Vector3(0, 0, 0),
            1, color, 0.15, 0.1
        );
        this.scene.add(arrow);
        
        const stateObj = { arrow, theta, phi, color, label };
        this.stateVectors.push(stateObj);
        
        if (label) {
            this.addLabel(label, x * 1.2, y * 1.2, z * 1.2, '#ffffff');
        }
        
        return stateObj;
    }
    
    updateStateVector(index, theta, phi) {
        if (index < this.stateVectors.length) {
            const state = this.stateVectors[index];
            const x = Math.sin(theta) * Math.cos(phi);
            const y = Math.sin(theta) * Math.sin(phi);
            const z = Math.cos(theta);
            state.arrow.setDirection(new THREE.Vector3(x, y, z).normalize());
            state.theta = theta;
            state.phi = phi;
        }
    }
    
    addRotationRing(axis, color = 0xffd93d) {
        const ringGeometry = new THREE.TorusGeometry(0.6, 0.02, 16, 100);
        const ringMaterial = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.5 });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        
        if (axis === 'x') ring.rotation.y = Math.PI / 2;
        else if (axis === 'y') ring.rotation.x = Math.PI / 2;
        
        this.scene.add(ring);
        return ring;
    }
    
    animateGate(gateName, duration = 1000) {
        const startTime = Date.now();
        const initialStates = this.stateVectors.map(s => ({ theta: s.theta, phi: s.phi }));
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const t = Math.min(elapsed / duration, 1);
            const easeT = 1 - Math.pow(1 - t, 3); // Ease out cubic
            
            this.stateVectors.forEach((state, i) => {
                const initial = initialStates[i];
                let newTheta = initial.theta;
                let newPhi = initial.phi;
                
                switch (gateName) {
                    case 'X':
                        newTheta = initial.theta + easeT * Math.PI;
                        break;
                    case 'Y':
                        newTheta = initial.theta + easeT * Math.PI;
                        newPhi = initial.phi + easeT * Math.PI;
                        break;
                    case 'Z':
                        newPhi = initial.phi + easeT * Math.PI;
                        break;
                    case 'H':
                        // Hadamard: complex rotation
                        const hTheta = Math.PI / 2;
                        newTheta = initial.theta + easeT * (hTheta - initial.theta);
                        newPhi = initial.phi + easeT * Math.PI;
                        break;
                    case 'S':
                        newPhi = initial.phi + easeT * Math.PI / 2;
                        break;
                    case 'T':
                        newPhi = initial.phi + easeT * Math.PI / 4;
                        break;
                }
                
                this.updateStateVector(i, newTheta, newPhi);
            });
            
            if (t < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.time += 0.01;
        
        // Run custom animations
        this.animations.forEach(anim => anim(this.time));
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    setupResize() {
        window.addEventListener('resize', () => {
            this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(this.container.clientWidth, this.options.height);
        });
    }
    
    addCustomAnimation(animFunc) {
        this.animations.push(animFunc);
    }
}

// Circuit Diagram Drawing
class QuantumCircuit {
    constructor(containerId, numQubits = 2, numGates = 5) {
        this.container = document.getElementById(containerId);
        this.numQubits = numQubits;
        this.numGates = numGates;
        this.gateWidth = 60;
        this.gateHeight = 40;
        this.qubitSpacing = 60;
        this.gateSpacing = 80;
        this.padding = 50;
        
        this.width = this.padding * 2 + this.numGates * this.gateSpacing;
        this.height = this.padding * 2 + (this.numQubits - 1) * this.qubitSpacing;
        
        this.createSVG();
        this.drawWires();
    }
    
    createSVG() {
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('width', this.width);
        this.svg.setAttribute('height', this.height);
        this.svg.style.background = '#0a0a0a';
        this.svg.style.borderRadius = '10px';
        this.container.appendChild(this.svg);
        
        // Add defs for gradients
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        
        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        gradient.setAttribute('id', 'gateGradient');
        gradient.innerHTML = `
            <stop offset="0%" style="stop-color:#667eea"/>
            <stop offset="100%" style="stop-color:#764ba2"/>
        `;
        defs.appendChild(gradient);
        
        this.svg.appendChild(defs);
    }
    
    drawWires() {
        for (let i = 0; i < this.numQubits; i++) {
            const y = this.padding + i * this.qubitSpacing;
            
            // Wire line
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', this.padding - 30);
            line.setAttribute('y1', y);
            line.setAttribute('x2', this.width - this.padding + 30);
            line.setAttribute('y2', y);
            line.setAttribute('stroke', '#64ffda');
            line.setAttribute('stroke-width', '2');
            this.svg.appendChild(line);
            
            // Qubit label
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', 15);
            label.setAttribute('y', y + 5);
            label.setAttribute('fill', '#64ffda');
            label.setAttribute('font-size', '14');
            label.setAttribute('font-family', 'monospace');
            label.textContent = `|q${i}⟩`;
            this.svg.appendChild(label);
        }
    }
    
    addGate(qubit, position, gateName, color = 'url(#gateGradient)') {
        const x = this.padding + position * this.gateSpacing;
        const y = this.padding + qubit * this.qubitSpacing;
        
        // Gate box
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x - this.gateWidth / 2);
        rect.setAttribute('y', y - this.gateHeight / 2);
        rect.setAttribute('width', this.gateWidth);
        rect.setAttribute('height', this.gateHeight);
        rect.setAttribute('rx', '8');
        rect.setAttribute('fill', color);
        rect.setAttribute('stroke', '#64ffda');
        rect.setAttribute('stroke-width', '2');
        this.svg.appendChild(rect);
        
        // Gate label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x);
        text.setAttribute('y', y + 5);
        text.setAttribute('fill', 'white');
        text.setAttribute('font-size', '16');
        text.setAttribute('font-weight', 'bold');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-family', 'monospace');
        text.textContent = gateName;
        this.svg.appendChild(text);
    }
    
    addCNOT(controlQubit, targetQubit, position) {
        const x = this.padding + position * this.gateSpacing;
        const cy = this.padding + controlQubit * this.qubitSpacing;
        const ty = this.padding + targetQubit * this.qubitSpacing;
        
        // Vertical line
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x);
        line.setAttribute('y1', cy);
        line.setAttribute('x2', x);
        line.setAttribute('y2', ty);
        line.setAttribute('stroke', '#ff6b6b');
        line.setAttribute('stroke-width', '2');
        this.svg.appendChild(line);
        
        // Control dot
        const control = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        control.setAttribute('cx', x);
        control.setAttribute('cy', cy);
        control.setAttribute('r', '8');
        control.setAttribute('fill', '#ff6b6b');
        this.svg.appendChild(control);
        
        // Target (XOR symbol)
        const target = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        target.setAttribute('cx', x);
        target.setAttribute('cy', ty);
        target.setAttribute('r', '15');
        target.setAttribute('fill', 'none');
        target.setAttribute('stroke', '#ff6b6b');
        target.setAttribute('stroke-width', '2');
        this.svg.appendChild(target);
        
        // Plus sign in target
        const hLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        hLine.setAttribute('x1', x - 15);
        hLine.setAttribute('y1', ty);
        hLine.setAttribute('x2', x + 15);
        hLine.setAttribute('y2', ty);
        hLine.setAttribute('stroke', '#ff6b6b');
        hLine.setAttribute('stroke-width', '2');
        this.svg.appendChild(hLine);
        
        const vLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        vLine.setAttribute('x1', x);
        vLine.setAttribute('y1', ty - 15);
        vLine.setAttribute('x2', x);
        vLine.setAttribute('y2', ty + 15);
        vLine.setAttribute('stroke', '#ff6b6b');
        vLine.setAttribute('stroke-width', '2');
        this.svg.appendChild(vLine);
    }
    
    addMeasurement(qubit, position) {
        const x = this.padding + position * this.gateSpacing;
        const y = this.padding + qubit * this.qubitSpacing;
        
        // Measurement box
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x - 25);
        rect.setAttribute('y', y - 20);
        rect.setAttribute('width', 50);
        rect.setAttribute('height', 40);
        rect.setAttribute('rx', '5');
        rect.setAttribute('fill', '#1a1a2e');
        rect.setAttribute('stroke', '#ffd93d');
        rect.setAttribute('stroke-width', '2');
        this.svg.appendChild(rect);
        
        // Meter arc
        const arc = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        arc.setAttribute('d', `M ${x - 15} ${y + 5} A 15 15 0 0 1 ${x + 15} ${y + 5}`);
        arc.setAttribute('fill', 'none');
        arc.setAttribute('stroke', '#ffd93d');
        arc.setAttribute('stroke-width', '2');
        this.svg.appendChild(arc);
        
        // Meter needle
        const needle = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        needle.setAttribute('x1', x);
        needle.setAttribute('y1', y + 5);
        needle.setAttribute('x2', x + 10);
        needle.setAttribute('y2', y - 10);
        needle.setAttribute('stroke', '#ffd93d');
        needle.setAttribute('stroke-width', '2');
        this.svg.appendChild(needle);
    }
}

// Matrix Visualization
function createMatrixVisualization(containerId, matrix, label = '') {
    const container = document.getElementById(containerId);
    const size = matrix.length;
    
    let html = `<div class="matrix-viz">`;
    if (label) {
        html += `<div class="matrix-label">${label} = </div>`;
    }
    html += `<div class="matrix-bracket">[</div>`;
    html += `<div class="matrix-content" style="grid-template-columns: repeat(${size}, 1fr);">`;
    
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const val = matrix[i][j];
            let displayVal = formatComplex(val);
            const magnitude = Math.sqrt(val.re * val.re + val.im * val.im);
            const hue = magnitude > 0.5 ? 160 : 0; // Green for large, red for small
            const lightness = 30 + magnitude * 40;
            
            html += `<div class="matrix-cell" style="background: hsl(${hue}, 70%, ${lightness}%)">${displayVal}</div>`;
        }
    }
    
    html += `</div>`;
    html += `<div class="matrix-bracket">]</div>`;
    html += `</div>`;
    
    container.innerHTML = html;
}

function formatComplex(val) {
    if (typeof val === 'number') {
        return val.toFixed(2);
    }
    const re = val.re || 0;
    const im = val.im || 0;
    
    if (Math.abs(im) < 0.001) return re.toFixed(2);
    if (Math.abs(re) < 0.001) {
        if (Math.abs(im - 1) < 0.001) return 'i';
        if (Math.abs(im + 1) < 0.001) return '-i';
        return im.toFixed(2) + 'i';
    }
    return `${re.toFixed(1)}${im >= 0 ? '+' : ''}${im.toFixed(1)}i`;
}

// State Vector Visualization
function createStateVisualization(containerId, amplitudes, labels = ['|0⟩', '|1⟩']) {
    const container = document.getElementById(containerId);
    
    let html = `<div class="state-viz">`;
    
    amplitudes.forEach((amp, i) => {
        const prob = amp.re * amp.re + amp.im * amp.im;
        const phase = Math.atan2(amp.im, amp.re);
        const phaseDeg = (phase * 180 / Math.PI).toFixed(0);
        
        html += `
            <div class="state-component">
                <div class="state-label">${labels[i]}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: ${prob * 100}%; background: linear-gradient(90deg, #64ffda, #4ecdc4);"></div>
                </div>
                <div class="state-info">
                    <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                    <span class="phase-value">φ = ${phaseDeg}°</span>
                </div>
            </div>
        `;
    });
    
    html += `</div>`;
    container.innerHTML = html;
}

// Export for use
window.BlochSphere = BlochSphere;
window.QuantumCircuit = QuantumCircuit;
window.createMatrixVisualization = createMatrixVisualization;
window.createStateVisualization = createStateVisualization;
