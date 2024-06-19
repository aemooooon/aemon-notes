<%* 
	const notes = app.vault.getMarkdownFiles(); 
	let output = ""; 
	notes.forEach(note => { 
		output += `- [[${note.basename}]]\n`; 
	}); 
	tR += output; 
%>